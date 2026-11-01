import torch
import torch.nn as nn
import torch.nn.functional as F
import math








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

        self.q_w = nn.Linear(d_model, q_heads * self.d_k, bias=False)

        self.k_w = nn.Linear(d_model, kv_heads * self.d_k, bias=False)

        self.v_w = nn.Linear(d_model, kv_heads * self.d_k, bias=False)

        self.o_w = nn.Linear(q_heads * self.d_k, d_model, bias=False)

        self.d_model = d_model

        self.init_std = init_std








    def forward(self, x, cache=None, dim=2):


        B, T, _ = x.shape


        Q = self.q_w(x).view(B, T, self.q_heads, self.d_k).transpose(1, 2)

        K = self.k_w(x).view(B, T, self.kv_heads, self.d_k).transpose(1, 2)

        V = self.v_w(x).view(B, T, self.kv_heads, self.d_k).transpose(1, 2)







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

        out = self.o_w(out)

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

        self.w1 = nn.Linear(d_model, d_ff, bias=False)

        self.w2 = nn.Linear(d_ff, d_model, bias=False)

        self.w3 = nn.Linear(d_model, d_ff, bias=False)


    def forward(self, x):

        return self.w2(F.silu(self.w1(x)) * self.w3(x))






class MoE(nn.Module):

    def __init__(self, d_model, d_ff, num_experts, K, alpha=0.01):

        super().__init__()

        self.d_model = d_model

        self.n_experts = num_experts

        self.K = K

        self.alpha = alpha


        self.gate = nn.Linear(d_model, num_experts, bias=False)

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



        gate_logits = self.gate(x_flat)

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


class ViTBlock(nn.Module):

    def __init__(self, d_model, q_heads, kv_heads, d_ff,

                 num_experts, K, alpha=0.01, eps=1e-6, init_std=0.02):

        super().__init__()

        self.norm1 = RMSNorm(d_model, eps=eps)

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


class ViT(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3,

                 n_classes=1000, d_model=768, n_layers=12,

                 q_heads=12, kv_heads=4, d_ff=3072,

                 num_experts=4, K=2, alpha=0.01, use_cls_token=True):

        super().__init__()

        self.use_cls_token = use_cls_token

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, d_model)

        n_patches = self.patch_embed.n_patches


        if use_cls_token:

            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

            nn.init.trunc_normal_(self.cls_token, std=0.02)

            pos_len = n_patches + 1

        else:

            pos_len = n_patches



        self.pos_embed = LearnedPositionalEncoding(pos_len, d_model)

        self.blocks = nn.ModuleList([

            ViTBlock(d_model, q_heads, kv_heads, d_ff,

                     num_experts, K, alpha=alpha)

            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)

        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):

        B = x.size(0)


        x = self.patch_embed(x)


        if self.use_cls_token:

            cls = self.cls_token.expand(B, -1, -1)

            x = torch.cat([cls, x], dim=1)

        x = self.pos_embed(x)

        aux_total = 0.0

        for blk in self.blocks:

            x = blk(x)

            aux_total = aux_total + blk.moe.aux_loss

        x = self.norm(x)



        if self.use_cls_token:

            feat = x[:, 0]

        else:

            feat = x.mean(dim=1)

        logits = self.head(feat)

        return logits, aux_total