# ViT-10M — Vision Transformer with MoE and Group Query Attention

A lightweight Vision Transformer (~10M parameters) for image classification on **ImageNet-100**, focused on architectural efficiency and training speed.

The model combines modern ideas such as **Group Query Attention (GQA)**, **Mixture of Experts (MoE)**, and **RMSNorm** to achieve competitive accuracy with a small parameter budget. Training fully supports **Distributed Data Parallel (DDP)** and **Mixed Precision (AMP)**.

---

## ✨ Features

### Model Architecture
- **Group Query Attention (GQA)**: Fewer Key/Value heads than Query heads for reduced memory and compute.
- **Mixture of Experts (MoE)**: Each ViT block uses multiple experts with Top-K gating instead of a single FFN.
- **RMSNorm**: A lighter, more stable alternative to LayerNorm.
- **Learned Positional Encoding**: Trainable positional embeddings with trunc-normal initialization.
- **CLS Token + Projection Head**: Follows the original ViT design for classification.
- **Auxiliary Load-Balancing Loss**: Keeps expert utilization balanced.

### Training
- Full **DDP** support (multi-GPU) with `DistributedSampler`.
- **Automatic Mixed Precision (AMP)** via `GradScaler`.
- **Cosine LR Schedule** with **Warmup**.
- **Mixup + CutMix + Label Smoothing** via `timm`.
- **Early Stopping** based on validation accuracy.
- **Automatic Resume** from the latest checkpoint.
- **Rolling checkpoint saving** (keeps the last N checkpoints).

### Dataset
- Loads **ImageNet-100** from HuggingFace (`clane9/imagenet-100`).
- Strong training augmentation: `RandomResizedCrop`, `RandAugment`, `RandomErasing`.
- Standard evaluation pipeline: `Resize → CenterCrop → Normalize`.

---

## 📊 Results

| Metric | Value |
|---|---|
| **Top-1 Accuracy @ Epoch 30** | **57%** |
| Top-5 Accuracy | Typically contains the correct class |

> **Note:** Due to the remaining ~43% error rate, the model often fails to predict the correct class as its **first** guess. However, the correct class is usually present within the **Top-5 predictions**, making Top-5 accuracy a more informative metric for this model at its current training stage.

---

## 📁 Project Structure

```
.
├── ViT_model.py                 # Model definition (ViT, MoE, GQA, RMSNorm, PatchEmbed)
├── Transformer_Dataloader.py    # Dataset and DataLoader for ImageNet-100
├── ViT_Train.py                 # Training loop + DDP + checkpointing
├── Ineferce_ViT_10m.py          # Inference script on a single image
└── ViT_10m.pt                   # Trained model weights
```

---

## ⚙️ Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install datasets pillow timm
```

> CUDA and NCCL must be available for DDP to work.

---

## 🚀 Usage

### 1) Train the Model

```bash
python ViT_Train.py
```

The script automatically detects the number of GPUs:
- If **2 or more** GPUs are available → trains with DDP across all GPUs.
- Otherwise → trains on a single GPU or CPU.

#### Key Configuration (inside `__main__` in `ViT_Train.py`)

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | `256` | Per-GPU batch size |
| `num_workers` | `4` | DataLoader workers |
| `epochs` | `200` | Number of epochs |
| `lr` | `3e-4` | Initial learning rate (AdamW) |
| `save_path` | `.../ViT_best.pt` | Best model save path |
| `save_every` | `2` | Save a checkpoint every N epochs |
| `keep_last_n` | `3` | Number of checkpoints to keep |
| `early_stop_acc` | `90.0` | Early-stopping accuracy threshold |
| `resume` | `True` | Resume from the latest checkpoint |

Checkpoints are saved to the first available directory among:
1. `/content/drive/MyDrive/ViT_10M_checkpoints`
2. `./ViT_10M_checkpoints`

### 2) Inference on a Single Image

```bash
python Ineferce_ViT_10m.py
```

Set the image path and checkpoint inside the script:

```python
CKPT_PATH = "ViT_10m.pt"
IMG_PATH  = "~/Pictures/Screenshots/testmodel2.png"
```

Example output:

```
Top-5 predictions:
  1. tabby, tabby cat        92.31%
  2. tiger cat               3.12%
  3. Egyptian cat            1.85%
  ...

Prediction: tabby, tabby cat  (92.31%)
```

---

## 🧠 Architecture Details

Model configuration (from `make_model` in `ViT_Train.py`):

| Hyperparameter | Value |
|---|---|
| `img_size` | 224 |
| `patch_size` | 16 |
| `d_model` | 224 |
| `n_layers` | 8 |
| `q_heads` | 8 |
| `kv_heads` | 2 |
| `d_ff` | 608 |
| `num_experts` | 4 |
| `K` (top-k routing) | 2 |
| `n_classes` | 100 |
| **Total Parameters** | **≈ 10M** |

### Group Query Attention
Instead of standard attention (where `q_heads == kv_heads`), each group of Query heads shares a single KV head. This:
- Reduces KV cache memory usage.
- Is slightly faster during training.
- Introduces negligible accuracy loss.

### Mixture of Experts
Each ViT block uses `num_experts` experts instead of a fixed FFN:
- **Gate**: A Linear layer producing softmax logits.
- **Top-K Routing**: Only the top `K` experts are activated per token.
- **Weighting**: Expert outputs are combined using softmax-normalized weights.
- **Aux Loss**: Prevents the gate from collapsing onto a single expert.

Final loss:

```
L_total = L_CE + aux_weight * L_aux
```

---

## 💾 Checkpoint Format

Periodic checkpoints contain:

```python
{
    "epoch": int,
    "model": state_dict,
    "optimizer": state_dict,
    "scheduler": state_dict,
    "scaler": state_dict,     # for AMP
    "best_val": float,
}
```

The final model (`ViT_best.pt`) contains only the model `state_dict`.

---

## 📊 Sample Training Log

```
epoch 00 | loss 4.3120 | train 5.21% | val 7.84% | lr 3.00e-06 | 142.3s
epoch 01 | loss 3.9870 | train 12.44% | val 15.02% | lr 6.10e-06 | 138.1s
   saved best (15.02%)
...
epoch 30 | loss 1.6420 | train 58.90% | val 57.00% | lr 2.11e-04 | 136.7s
```

---

## 📝 Notes

- For DDP, port `29500` must be free (configurable via `MASTER_PORT`).
- Under DDP, only rank 0 saves checkpoints and prints logs.
- When early stopping triggers, the decision is broadcast to all ranks.
- The validation DataLoader uses `num_workers/2` to avoid competing with training.

---

## ⚠️ Known Limitations

- At ~30 epochs of training, the model reaches **~57% Top-1 accuracy** on ImageNet-100.
- The remaining **~43% error** means the model frequently misses the correct class as its first prediction, but the correct class is usually present within the **Top-5** predictions.
- Further gains are expected with longer training, more aggressive augmentation, or larger expert capacity.

---

## 📜 License

This project is released under the MIT License — free to use and modify.

---

## 🙏 References

- [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [Switch Transformers (MoE)](https://arxiv.org/abs/2101.03961)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- Dataset: [`clane9/imagenet-100`](https://huggingface.co/datasets/clane9/imagenet-100)