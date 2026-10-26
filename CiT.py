import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from megatron.core.tensor_parallel import(
    ColumnParallelLinear as CLinear,
    RowParallelLinear as RLinear,
)

from megatron.core.transformer import TransformerConfig
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
import os





def build_config(tp_size: int = 1) -> TransformerConfig:

    cfg = TransformerConfig(

        num_layers=12,

        hidden_size=768,

        num_attention_heads=12,

        num_query_groups=4,

    
        ffn_hidden_size=3072,

        num_moe_experts=4,

        moe_ffn_hidden_size=3072,

        moe_router_topk=2,

        moe_router_load_balancing_type="aux_loss",

        moe_aux_loss_coeff=0.01, 

        moe_token_dispatcher_type="allgather",

        moe_grouped_gemm=False,


        
        normalization="RMSNorm",

        layernorm_epsilon=1e-6,

        layernorm_zero_centered_gamma=False,

        
        add_bias_linear=False,

        add_qkv_bias=False,

        
        activation_func=F.silu,

        gated_linear_unit=True,


        params_dtype=torch.float16,

        bf16=False,

        init_method_std=0.02,

        
        tensor_model_parallel_size=tp_size,

        pipeline_model_parallel_size=1,

        expert_model_parallel_size=1,

        sequence_parallel=False,

    
    )

    return cfg

__CFG__CACHE = {}


def init_distributed(tp_size: int = 1):

    tp_size = int(tp_size)

    if tp_size in __CFG__CACHE:

        return __CFG__CACHE[tp_size] 

    if not dist.is_initialized():

        dist.init_process_group("nccl") #err

    mpu.initialize_model_parallel(tensor_model_parallel_size=tp_size)

    model_parallel_cuda_manual_seed(1234)

    cfg = build_config(tp_size)

    __CFG__CACHE[tp_size] = cfg

    return cfg







class Cnn(nn.Module):

    def __init__(self, in_channels: int, padding: int,
                 
                num_classes: int, stride: int, kernel_size: int):

        super().__init__()

        self.model = nn.Sequential(

            nn.Conv2d(in_channels, 64, kernel_size, stride, padding, dtype=torch.float16, bias=False),

            nn.GroupNorm(num_groups=32, num_channels=64),
            

            nn.GLU(dim=1),

            nn.Conv2d(32, 64, kernel_size, padding=padding, dtype=torch.float16, bias=False),

            nn.GroupNorm(num_groups=32, num_channels=64),

            nn.GLU(dim=1),

            nn.MaxPool2d((2, 2), ceil_mode=True),



            nn.Conv2d(32, 128, kernel_size, padding=padding, dtype=torch.float16, bias=False),

            nn.GroupNorm(num_groups=32, num_channels=128),
            

            nn.GLU(dim=1),

            nn.MaxPool2d((2, 2), ceil_mode=True),



            nn.Conv2d(64, 128, kernel_size, padding=padding, dtype=torch.float16, bias=False),

            nn.GroupNorm(num_groups=32, num_channels=128),
            

            nn.GLU(dim=1),

            nn.MaxPool2d((2, 2), ceil_mode=True),



            nn.Conv2d(64, 256, kernel_size, padding=padding, dtype=torch.float16, bias=False),

            nn.GroupNorm(num_groups=32, num_channels=256),
            

            nn.GLU(dim=1),

            nn.MaxPool2d((2, 2), ceil_mode=True),



            nn.Conv2d(128, 256, kernel_size, padding=padding, dtype=torch.float16, bias=False),

            nn.GroupNorm(num_groups=32, num_channels=256),
            

            nn.GLU(dim=1),

            nn.MaxPool2d((2, 2), ceil_mode=True),



            nn.Conv2d(128, 512, kernel_size, padding=padding, dtype=torch.float16, bias=False),

            nn.GroupNorm(num_groups=32, num_channels=512),
            

            nn.GLU(dim=1),

            nn.MaxPool2d((2, 2), ceil_mode=True),



            nn.Conv2d(256, 512, kernel_size, padding=padding, dtype=torch.float16, bias=False),

            nn.GroupNorm(num_groups=32, num_channels=512),
            

            nn.GLU(dim=1)
        )




    def forward(self, x):

        if x.dtype != torch.float16:

            x = x.half()

        return self.model(x)







class LearnedPositionalEncoding(nn.Module):



    def __init__(self, max_len: int, d_model: int, init_std: float = 0.02):

        super().__init__()

        self.max_len = max_len

        self.d_model = d_model




        self.pos_embed = nn.Parameter(torch.zeros((1, max_len, d_model)))

        nn.init.trunc_normal_(self.pos_embed, std=init_std)



    def forward(self, x):



        N = x.size(1)

        assert N <= self.max_len, \
            f"sequence length {N} > max_len {self.max_len}"

        return x + self.pos_embed[:, :N, :]









class GroupQueryAttention(nn.Module):

    def __init__(self, q_heads: int, kv_heads: int, d_model: int, init_std: int):

        super().__init__()

        assert q_heads % kv_heads == 0

        assert d_model % q_heads == 0

        self.d_model = d_model

        self.q_heads = q_heads

        self.kv_heads = kv_heads

        self.d_k = d_model // q_heads

        self.n_rep = q_heads // kv_heads

        config = init_distributed(tp_size=1) #err

        self.q_w = CLinear(d_model, q_heads * self.d_k, gather_output=False, config=config, init_method=nn.init.xavier_uniform_, skip_bias_add=False, bias=False)

        self.k_w = CLinear(d_model, kv_heads * self.d_k, gather_output=False, config=config, init_method=nn.init.xavier_uniform_, skip_bias_add=False, bias=False)

        self.v_w = CLinear(d_model, kv_heads * self.d_k, gather_output=False, config=config, init_method=nn.init.xavier_uniform_, skip_bias_add=False, bias=False)

        self.o_w = RLinear(q_heads * self.d_k, d_model, input_is_parallel=True, config=config, init_method=nn.init.xavier_uniform_, skip_bias_add=False, bias=False)

        self.d_model = d_model

        self.init_std = init_std








    def forward(self, x, cache=None, dim=2):


        B, T, _ = x.shape


        Q, _ = self.q_w(x)

        Q = Q.view(B, T, self.q_heads, self.d_k).transpose(1, 2)

        K, _ = self.k_w(x)

        K = K.view(B, T, self.kv_heads, self.d_k).transpose(1, 2)

        V, _ = self.v_w(x)

        V = V.view(B, T, self.kv_heads, self.d_k).transpose(1, 2)





        if cache is not None:


            K_prev, V_prev = cache

            K = torch.cat([K_prev, K], dim=dim)

            V = torch.cat([V_prev, V], dim=dim)


        new_cache = (K, V)




        if self.n_rep > 1:

            K = K.repeat_interleave(self.n_rep, dim=1)

            V = V.repeat_interleave(self.n_rep, dim=1)




        scale = 1.0 / math.sqrt(self.d_k)

        score = (Q @ K.transpose(-1, -2)) * scale

        attn = F.softmax(score, dim=-1)

        out = attn @ V


        out = out.transpose(1, 2).contiguous().view(B, T, self.q_heads * self.d_k)

        out, _ = self.o_w(out)

        return out, new_cache










class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):

        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.ones(dim))



    def forward(self, x):


        dtype = x.dtype

        x_f = x.float()

        rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        return (x_f * rms).to(dtype) * self.weight



if __name__ == "__main__":

    rmsnorm = RMSNorm()

    torch.compile(rmsnorm)









class Expert(nn.Module):

    def __init__(self, d_model, d_ff):

        super().__init__()

        config = init_distributed(tp_size=True)

        self.w1 = CLinear(d_model, d_ff, gather_output=False, config=config, init_method=nn.init.xavier_uniform_, skip_bias_add=False, bias=False) #this wey
 
        self.w2 = RLinear(d_ff, d_model, config=config, init_method=nn.init.xavier_uniform_, skip_bias_add=False, input_is_parallel=True, bias=False) #this wey

        self.w3 = CLinear(d_model, d_ff, config=config, init_method=nn.init.xavier_uniform_, skip_bias_add=False, bias=False) #this wey


    def forward(self, x):

        w1, _ = self.w1(x)

        w3, _ = self.w3(x)

        mid = F.silu(w1) * w3

        out, _ = self.w2(mid)

        return out






class MoE(nn.Module):

    def __init__(self, d_model, d_ff, num_experts, K, alpha=0.01):

        super().__init__()

        self.d_model = d_model

        self.n_experts = num_experts

        self.K = K

        self.alpha = alpha

        config = init_distributed(tp_size=1)


        self.gate = CLinear(d_model, num_experts, gather_output=True, config=config, init_method=nn.init.xavier_uniform_, skip_bias_add=False, bias=False)  #this wey

        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)
        ])


        self.aux_loss = torch.tensor(0.0)


    def forward(self, x):

        if x.dim() == 3:

            B, T, D = x.shape

            x_flat = x.reshape(B * T, D)

            restore_shape = True

        elif x.dim() == 2:

            x_flat = x

            restore_shape = False



        N, D = x_flat.shape



        gate_logits, _ = self.gate(x_flat)

        gate_probs = F.softmax(gate_logits, dim=-1)


        top_k, top_idx = torch.topk(gate_probs, self.K, dim=-1)

        top_k = top_k / top_k.sum(dim=-1, keepdim=True)


        with torch.no_grad():

            one_hot = F.one_hot(top_idx, num_classes=self.n_experts).float()

            tokens_per_expert = one_hot.sum(dim=(0, 1))

            f = tokens_per_expert / N


        P = gate_probs.mean(dim=0)


        self.aux_loss = self.alpha * self.n_experts * torch.sum(f * P)



        out = torch.zeros_like(x_flat)

        flat_idx = top_idx.reshape(-1)

        flat_k = top_k.reshape(-1)

        flat_tokens = torch.arange(N, device=x.device).repeat_interleave(self.K)


        for i, expert in enumerate(self.experts):

            sel = (flat_idx == i)

            if not sel.any():

                continue

            token_ids = flat_tokens[sel]

            weights = flat_k[sel]


            y = expert(x_flat[token_ids])

            out.index_add_(0, token_ids, weights.unsqueeze(-1) * y)

        if restore_shape:

            out = out.reshape(B, T, D)


        return out
















class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, d_model=768):

        super().__init__()

        assert img_size % patch_size == 0

        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(

            in_chans, d_model,

            kernel_size=patch_size, stride=patch_size,

        )


    def forward(self, x):


        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)

        return x






class CiTBlock(nn.Module):

    def __init__(self, d_model, q_heads, kv_heads, d_ff,

                 num_experts, K, in_channels, padding, kernel_size, stride, alpha=0.01, eps=1e-6, init_std=0.02):

        super().__init__()

        
        self.norm1 = RMSNorm(d_model, eps=eps)

        #err

        self.attn = GroupQueryAttention(

            q_heads=q_heads, kv_heads=kv_heads, d_model=d_model, init_std=init_std


        )

        self.norm2 = RMSNorm(d_model, eps=eps)

        self.moe = MoE(d_model, d_ff, num_experts, K, alpha=alpha)

    def forward(self, x):



        h, _ = self.attn(self.norm1(x))

        x = x + h

        x = x + self.moe(self.norm2(x))

        return x




class CiT(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 
                 num_classes=1000, d_model=768, n_layers=12,

                 q_heads=12, kv_heads=4, d_ff=3072,

                 num_experts=4, K=2, alpha=0.01,

                 in_channels=3, kernel_size=3, stride=1, padding=1,

                 use_cls_token=True, dropout=0.1):
        
        super().__init__()

        self.use_cls_token = use_cls_token


        
        
        self.cnn = Cnn(in_channels=in_channels, padding=padding,
                       
                       kernel_size=kernel_size, stride=stride, num_classes=num_classes)
        
        

        
        self.patch_embed = nn.Conv2d(256, d_model, kernel_size=1)

        
        self.pos_embed = LearnedPositionalEncoding(2000, d_model) 

        if use_cls_token:

            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

            nn.init.trunc_normal_(self.cls_token, std=0.02)

        
        
        self.blocks = nn.ModuleList([

            # err
            CiTBlock(d_model, q_heads, kv_heads, d_ff,
                     
                     num_experts, K, alpha=alpha,

                     in_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding)

            for _ in range(n_layers)

        ])

        self.norm = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        
        B, C, H, W = x.shape

    
        x = self.cnn(x)

        B, Cc, Hp, Wp = x.shape

        
        

        x = x.reshape(B , Cc, Hp, Wp)

        
        x = self.patch_embed(x)

        x = x.flatten(2).transpose(1, 2)

        

    
        

        cls = self.cls_token.expand(B, -1, -1)

        x = torch.cat([cls, x], dim=1)

        x = self.pos_embed(x)


        aux_total = 0.0

        for blk in self.blocks:

            x = blk(x)

            aux_total = aux_total + blk.moe.aux_loss


        x = self.norm(x)

        feat = x[:, 0] if self.use_cls_token else x.mean(dim=1)

        logits = self.head(self.dropout(feat))

        return logits, aux_total

