# CiT: A Hybrid Vision Transformer with Mixture of Experts

An experimental vision architecture combining CNN feature extraction, Grouped-Query Attention (GQA), and Mixture of Experts (MoE) with tensor parallelism (via Megatron-Core), trained on ImageNet-100 using DeepSpeed.

This is a research-oriented implementation exploring hybrid CNN-Transformer designs with sparse expert routing for vision tasks.

---

## Overview

| Item | Value |
|---|---|
| **Model** | CiT (CNN + Transformer + MoE) |
| **Parameters** | ~10M (ViT-10M variant) |
| **Dataset** | ImageNet-100 (via HuggingFace `clane9/imagenet-100`) |
| **Training framework** | DeepSpeed (ZeRO-2) + Megatron-Core |
| **Precision** | Mixed precision (FP16) |
| **Task** | Image classification (100 classes) |

---

## Architecture

The model is built from the following components:

- **CNN Backbone** — A series of `Conv2d + GroupNorm + GLU + MaxPool2d` layers that extract local, hierarchical visual features before the transformer.
- **Patch Embedding** — A `1x1 Conv2d` projects the CNN feature map into the transformer hidden dimension.
- **Grouped-Query Attention (GQA)** — Query heads are split into groups; each group shares key/value heads (e.g., 8 Q heads, 2 KV heads). This reduces KV cache and memory usage.
- **Mixture of Experts (MoE)** — Each transformer block contains a router (`Linear`) and 4 experts with `top-K=2` routing, plus an auxiliary load-balancing loss (`aux_loss`).
- **RMSNorm** — Used instead of LayerNorm for better stability in mixed precision.
- **Learned Positional Encoding** — A trainable positional embedding (max_len=2000).
- **CLS Token** — An optional learnable `[CLS]` token is prepended and used for classification.

---

## Training Setup

- **Optimizer**: AdamW (`lr=2.1e-4`, `betas=(0.9, 0.999)`, `weight_decay=0.05`)
- **Scheduler**: `WarmupCosineLR` (warmup=1005 steps, total=15000 steps, `cos_min_ratio=0.0001`)
- **Batch size**: 128 (achieved via `train_micro_batch_size=2` × `gradient_accumulation_steps=64`)
- **Gradient clipping**: 1.0
- **Mixed precision**: FP16 with dynamic loss scaling
- **DeepSpeed ZeRO-2** with communication overlap for memory efficiency
- **Data augmentation**: `RandomResizedCrop`, `RandomHorizontalFlip`, `RandAugment`, `RandomErasing`, `Mixup` + `CutMix`

---

## Training Progress (from `epochs.txt`)

| Epoch | Loss | Train Acc | Val Acc |
|---|---|---|---|
| 0 | 4.69 | — | 2.44% |
| 1 | 4.53 | — | 5.88% |
| 5 | 4.08 | — | 17.26% |
| 8 | 3.94 | — | 20.80% |
| 10 | 3.84 | — | 23.60% |

Training was run for at least 10 epochs and val accuracy was still climbing (~2.3% gain per epoch). Best checkpoint was saved as `best/model.pt` in the checkpoint directory.

---

## Project Structure

- `CiT.py` — Model definition (CNN, GQA, MoE, CiTBlock, CiT)
- `CiT_Train.py` — Training loop, checkpointing, DeepSpeed integration
- `Transformer_Dataloader.py` — ImageNet-100 data loading with distributed samplers
- `ds_config.json` — DeepSpeed configuration (ZeRO-2, scheduler, optimizer)
- `requirements.txt`
- `README.md`

---

## Installation

    git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
    cd YOUR_REPO

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

### requirements.txt

    torch>=2.1.0
    torchvision>=0.16.0
    deepspeed>=0.13.0
    megatron-core>=0.6.0
    datasets>=2.18.0
    timm>=0.9.0
    pillow>=10.0.0

---

## How to Run Training

Single GPU or multi-GPU with DeepSpeed:

    deepspeed --num_gpus=1 CiT_Train.py --deepspeed ds_config.json --epochs 200 --batch_size 128

Or using a wrapper:

    deepspeed --num_gpus=<N> CiT_Train.py \
        --deepspeed ds_config.json \
        --epochs 200 \
        --batch_size 128 \
        --lr 2.1e-4 \
        --resume

Key arguments:
- `--epochs` — number of training epochs (default: 200)
- `--batch_size` — global batch size (default: 64)
- `--lr` — learning rate (default: 3e-4)
- `--resume` — resume from latest checkpoint
- `--ckpt_dir` — checkpoint directory (auto-selected if not provided)
- `--early_stop_acc` — early stopping threshold (default: 90.0%)

---

## Notes & Known Limitations

- **Small model size.** The default config uses `d_model=224, n_layers=8, d_ff=608` (~10M parameters). Validation accuracy is expected to be much lower than standard ViT-Base.
- **Experimental MoE.** The MoE routing uses `allgather` token dispatcher and a manual `index_add_` loop over experts, which is correct but not the most efficient implementation for larger models.
- **Tensor parallelism via Megatron-Core.** Requires `initialize_model_parallel` and a distributed environment; single-GPU usage may need adjustment.
- **Training time.** Each epoch takes ~15 minutes on a single GPU (based on `epochs.txt`).
- **Not yet converged.** Training is ongoing; results shown are after only 10 epochs.

---

## Future Work

- Scale up to `d_model=384` or `512` with more layers.
- Replace manual expert loop with `megatron.core.transformer.moe` dispatcher for better throughput.
- Try BF16 instead of FP16 for better stability.
- Add gradient checkpointing for larger variants.
- Evaluate on full ImageNet-1K.

---

## License

MIT — free to use, modify, and distribute.

## Acknowledgements

- [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) — tensor parallelism primitives
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) — ZeRO optimizer
- [HuggingFace Datasets](https://huggingface.co/datasets/clane9/imagenet-100) — ImageNet-100
- [timm](https://github.com/huggingface/pytorch-image-models) — Mixup / CutMix