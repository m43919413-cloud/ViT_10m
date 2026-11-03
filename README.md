```markdown
# ViT-MoE

A compact Vision Transformer (~10M parameters) combining **Grouped Query Attention (GQA)** with a **Mixture of Experts (MoE)** feed-forward, trained on ImageNet-100.

> ⚠️ **Note:** This model was trained for only **20 epochs** on a small subset (ImageNet-100), so its classification accuracy is low and predictions are not reliable. It is intended as a learning/research artifact — not a production classifier. See [Roadmap & Call for Contributions](#roadmap--call-for-contributions) if you'd like to help improve it.

---

## Overview

This repository contains a compact ViT implementation with several efficiency-oriented design choices:

- **Grouped Query Attention (GQA)** — fewer KV heads than Q heads, reducing KV cache and compute.
- **Mixture of Experts (MoE)** — 4 SwiGLU experts per block, top-2 routing with auxiliary load-balancing loss.
- **RMSNorm** — instead of LayerNorm.
- **Learned positional embeddings** — no sin/cos, no RoPE.
- **Optional CLS token** — or global average pooling.
- **Automatic DDP** — multi-GPU out of the box.
- **Automatic config inference** — inference script reads model hyperparameters directly from the checkpoint.

---

## Architecture

| Component | Value |
|---|---|
| Image size | 224 × 224 |
| Patch size | 16 × 16 → 196 patches |
| `d_model` | 224 |
| Layers | 8 |
| Q heads / KV heads | 8 / 2 (GQA, group size 4) |
| `d_ff` (per expert) | 608 |
| Experts per MoE layer | 4 |
| Top-K routing | 2 |
| Classes | 100 (ImageNet-100) |
| Total parameters | ≈ 10M |

Each MoE expert is a **SwiGLU** MLP:

```python
expert(x) = w2( SiLU(w1(x)) * w3(x) )
```

Routing is a linear gate `d_model → n_experts` → softmax → top-K → renormalize. A Switch-Transformer-style auxiliary load-balancing loss is added to the total loss during training.

---

## Repository Structure

| File | Description |
|---|---|
| `ViT_model.py` | Model: `ViT`, `ViTBlock`, `GroupQueryAttention`, `MoE`, `Expert`, `RMSNorm`, `PatchEmbed`, `LearnedPositionalEncoding` |
| `ViT_Train.py` | Training loop — AMP, Mixup/CutMix, cosine LR + warmup, DDP, checkpoint resume, early stopping |
| `Transformer_Dataloader.py` | ImageNet-100 dataloader (`clane9/imagenet-100`) with RandAugment + RandomErasing |
| `Inference_ViT.py` | Inference with **auto-config detection** from the checkpoint state_dict |
| `class_names.json` | 100 ImageNet-100 class names |

---

## Installation

```bash
pip install torch torchvision timm datasets pillow
```

The dataset is downloaded automatically from HuggingFace (`clane9/imagenet-100`).

---

## Training Pipeline

```bash
python ViT_Train.py
```

Default config (`__main__` in `ViT_Train.py`):

```python
cfg = dict(
    batch_size=256,
    num_workers=4,
    epochs=200,
    lr=3e-4,
    save_every=2,
    keep_last_n=3,
    early_stop_acc=90.0,
    resume=True,
)
```

### Pipeline features

1. **Data loading** (`Transformer_Dataloader.py`)
   - Train: `RandomResizedCrop(scale=(0.35, 1.0))`, `RandomHorizontalFlip`, `RandAugment(2, 7)`, `RandomErasing(p=0.25)`.
   - Val: `Resize(256)` + `CenterCrop(224)`.
   - ImageNet normalization; RGB conversion; HuggingFace Arrow → PIL → tensor.

2. **Distributed training**
   - If `torch.cuda.device_count() >= 2`, launches DDP via `mp.spawn` with `nccl` backend.
   - Otherwise falls back to single-GPU / CPU.
   - `DistributedSampler` with `set_epoch()` per epoch for correct shuffling.

3. **Mixed precision**
   - `torch.amp.autocast("cuda")` + `GradScaler`.

4. **Regularization**
   - Mixup (`alpha=0.2`) + CutMix (`alpha=1.0`) via `timm.data.mixup` with `prob=0.5`, `label_smoothing=0.1`.
   - Gradient clipping (`clip=1.0`).
   - Weight decay `0.05` on AdamW.

5. **LR schedule**
   - Linear warmup (`warmup=1000` steps) + cosine decay to 0.

6. **Checkpointing**
   - Best model saved to `save_path`.
   - Periodic checkpoint every `save_every` epochs → `ckpt_epoch_XXX.pt`, keeping the last `keep_last_n`.
   - Full state saved: model, optimizer, scheduler, scaler, best_val, epoch.
   - **Auto-resume** from the latest checkpoint on restart.

7. **Early stopping**
   - Stops when `val_acc >= early_stop_acc`.
   - Broadcast flag across DDP ranks.

8. **Metrics**
   - Per-epoch train loss/acc, val acc, current LR, wall time.
   - All metrics all-reduced across ranks in DDP.

---

## Inference Pipeline

```bash
python Inference_ViT.py --ckpt path/to/ViT_best.pt --image path/to/image.jpg --topk 5
```

The inference script **infers the architecture directly from the checkpoint state_dict** — `d_model`, `n_layers`, `q_heads`/`kv_heads`, `d_ff`, `num_experts`, `patch_size`, `img_size`, `use_cls_token`, `n_classes` — so you don't need to hand-specify hyperparameters.

- Handles `module.` prefixes (DDP-saved checkpoints) automatically.
- Handles both raw `state_dict` and `{"model": ..., ...}` checkpoint dicts.
- Warns on missing/unexpected keys and on head-config ambiguity.
- Manual overrides available via `--d_model`, `--n_layers`, `--q_heads`, etc.

Example:

```
path/to/image.jpg
  #1: tabby, tabby cat | class 281 | prob 0.3421
  #2: Egyptian cat     | class 285 | prob 0.2130
  ...
```

---

## Roadmap & Call for Contributions

This model is currently trained on **ImageNet-100 for 20 epochs** — far from what a ViT needs to be competitive. If you have access to strong GPUs and would like to help, contributions are very welcome:

- 🔁 **Retrain on ImageNet-1k** — the biggest single improvement.
- 📈 **Scale up the model** — increase `d_model`, `n_layers`, and `d_ff` (e.g., `d_model=384`, `n_layers=12`), while staying within your compute budget.
- ⏱️ **Train longer** — ViTs typically need 100–300+ epochs from scratch, or use a distillation/pretrained-init recipe.
- 🧪 **Ablations** — GQA group size, number of experts, top-K, aux-loss weight.
- 📊 **Evaluation** — top-1 / top-5 on the full val set, and comparisons to a same-size vanilla ViT.

**How to contribute:**

1. Fork this repo.
2. Train on your hardware (ImageNet-1k preferred, larger model preferred).
3. Open a **Pull Request** with:
   - the updated config (and any code changes),
   - final top-1 / top-5 numbers,
   - the checkpoint (or a link to it) if possible,
   - a short training log summary (epochs, batch size, hardware).

Even a partial improvement (e.g., 100 epochs on ImageNet-100 with a bigger model) is useful — please share it.

---

## Limitations

- Trained for only 20 epochs on ImageNet-100 → **low accuracy, unreliable predictions**. Do not use for real-world decisions.
- ~10M parameters — much smaller than ViT-B/16 (~86M).
- No public pretrained weights; trained from scratch.

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
``````markdown
# ViT-MoE

A compact Vision Transformer (~10M parameters) combining **Grouped Query Attention (GQA)** with a **Mixture of Experts (MoE)** feed-forward, trained on ImageNet-100.

> ⚠️ **Note:** This model was trained for only **20 epochs** on a small subset (ImageNet-100), so its classification accuracy is low and predictions are not reliable. It is intended as a learning/research artifact — not a production classifier. See [Roadmap & Call for Contributions](#roadmap--call-for-contributions) if you'd like to help improve it.

---

## Overview

This repository contains a compact ViT implementation with several efficiency-oriented design choices:

- **Grouped Query Attention (GQA)** — fewer KV heads than Q heads, reducing KV cache and compute.
- **Mixture of Experts (MoE)** — 4 SwiGLU experts per block, top-2 routing with auxiliary load-balancing loss.
- **RMSNorm** — instead of LayerNorm.
- **Learned positional embeddings** — no sin/cos, no RoPE.
- **Optional CLS token** — or global average pooling.
- **Automatic DDP** — multi-GPU out of the box.
- **Automatic config inference** — inference script reads model hyperparameters directly from the checkpoint.

---

## Architecture

| Component | Value |
|---|---|
| Image size | 224 × 224 |
| Patch size | 16 × 16 → 196 patches |
| `d_model` | 224 |
| Layers | 8 |
| Q heads / KV heads | 8 / 2 (GQA, group size 4) |
| `d_ff` (per expert) | 608 |
| Experts per MoE layer | 4 |
| Top-K routing | 2 |
| Classes | 100 (ImageNet-100) |
| Total parameters | ≈ 10M |

Each MoE expert is a **SwiGLU** MLP:

```python
expert(x) = w2( SiLU(w1(x)) * w3(x) )
```

Routing is a linear gate `d_model → n_experts` → softmax → top-K → renormalize. A Switch-Transformer-style auxiliary load-balancing loss is added to the total loss during training.

---

## Repository Structure

| File | Description |
|---|---|
| `ViT_model.py` | Model: `ViT`, `ViTBlock`, `GroupQueryAttention`, `MoE`, `Expert`, `RMSNorm`, `PatchEmbed`, `LearnedPositionalEncoding` |
| `ViT_Train.py` | Training loop — AMP, Mixup/CutMix, cosine LR + warmup, DDP, checkpoint resume, early stopping |
| `Transformer_Dataloader.py` | ImageNet-100 dataloader (`clane9/imagenet-100`) with RandAugment + RandomErasing |
| `Inference_ViT.py` | Inference with **auto-config detection** from the checkpoint state_dict |
| `class_names.json` | 100 ImageNet-100 class names |

---

## Installation

```bash
pip install torch torchvision timm datasets pillow
```

The dataset is downloaded automatically from HuggingFace (`clane9/imagenet-100`).

---

## Training Pipeline

```bash
python ViT_Train.py
```

Default config (`__main__` in `ViT_Train.py`):

```python
cfg = dict(
    batch_size=256,
    num_workers=4,
    epochs=200,
    lr=3e-4,
    save_every=2,
    keep_last_n=3,
    early_stop_acc=90.0,
    resume=True,
)
```

### Pipeline features

1. **Data loading** (`Transformer_Dataloader.py`)
   - Train: `RandomResizedCrop(scale=(0.35, 1.0))`, `RandomHorizontalFlip`, `RandAugment(2, 7)`, `RandomErasing(p=0.25)`.
   - Val: `Resize(256)` + `CenterCrop(224)`.
   - ImageNet normalization; RGB conversion; HuggingFace Arrow → PIL → tensor.

2. **Distributed training**
   - If `torch.cuda.device_count() >= 2`, launches DDP via `mp.spawn` with `nccl` backend.
   - Otherwise falls back to single-GPU / CPU.
   - `DistributedSampler` with `set_epoch()` per epoch for correct shuffling.

3. **Mixed precision**
   - `torch.amp.autocast("cuda")` + `GradScaler`.

4. **Regularization**
   - Mixup (`alpha=0.2`) + CutMix (`alpha=1.0`) via `timm.data.mixup` with `prob=0.5`, `label_smoothing=0.1`.
   - Gradient clipping (`clip=1.0`).
   - Weight decay `0.05` on AdamW.

5. **LR schedule**
   - Linear warmup (`warmup=1000` steps) + cosine decay to 0.

6. **Checkpointing**
   - Best model saved to `save_path`.
   - Periodic checkpoint every `save_every` epochs → `ckpt_epoch_XXX.pt`, keeping the last `keep_last_n`.
   - Full state saved: model, optimizer, scheduler, scaler, best_val, epoch.
   - **Auto-resume** from the latest checkpoint on restart.

7. **Early stopping**
   - Stops when `val_acc >= early_stop_acc`.
   - Broadcast flag across DDP ranks.

8. **Metrics**
   - Per-epoch train loss/acc, val acc, current LR, wall time.
   - All metrics all-reduced across ranks in DDP.

---

## Inference Pipeline

```bash
python Inference_ViT.py --ckpt path/to/ViT_best.pt --image path/to/image.jpg --topk 5
```

The inference script **infers the architecture directly from the checkpoint state_dict** — `d_model`, `n_layers`, `q_heads`/`kv_heads`, `d_ff`, `num_experts`, `patch_size`, `img_size`, `use_cls_token`, `n_classes` — so you don't need to hand-specify hyperparameters.

- Handles `module.` prefixes (DDP-saved checkpoints) automatically.
- Handles both raw `state_dict` and `{"model": ..., ...}` checkpoint dicts.
- Warns on missing/unexpected keys and on head-config ambiguity.
- Manual overrides available via `--d_model`, `--n_layers`, `--q_heads`, etc.

Example:

```
path/to/image.jpg
  #1: tabby, tabby cat | class 281 | prob 0.3421
  #2: Egyptian cat     | class 285 | prob 0.2130
  ...
```

---

## Roadmap & Call for Contributions

This model is currently trained on **ImageNet-100 for 20 epochs** — far from what a ViT needs to be competitive. If you have access to strong GPUs and would like to help, contributions are very welcome:

- 🔁 **Retrain on ImageNet-1k** — the biggest single improvement.
- 📈 **Scale up the model** — increase `d_model`, `n_layers`, and `d_ff` (e.g., `d_model=384`, `n_layers=12`), while staying within your compute budget.
- ⏱️ **Train longer** — ViTs typically need 100–300+ epochs from scratch, or use a distillation/pretrained-init recipe.
- 🧪 **Ablations** — GQA group size, number of experts, top-K, aux-loss weight.
- 📊 **Evaluation** — top-1 / top-5 on the full val set, and comparisons to a same-size vanilla ViT.

**How to contribute:**

1. Fork this repo.
2. Train on your hardware (ImageNet-1k preferred, larger model preferred).
3. Open a **Pull Request** with:
   - the updated config (and any code changes),
   - final top-1 / top-5 numbers,
   - the checkpoint (or a link to it) if possible,
   - a short training log summary (epochs, batch size, hardware).

Even a partial improvement (e.g., 100 epochs on ImageNet-100 with a bigger model) is useful — please share it.

---

## Limitations

- Trained for only 20 epochs on ImageNet-100 → **low accuracy, unreliable predictions**. Do not use for real-world decisions.
- ~10M parameters — much smaller than ViT-B/16 (~86M).
- No public pretrained weights; trained from scratch.

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
```