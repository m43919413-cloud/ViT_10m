# inference.py
import os
import json
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from ViT_model import ViT

try:
    from Transformer_Dataloader import MEAN, STD
    _HAS_DL = True
except Exception:
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    _HAS_DL = False

CLASS_NAMES_JSON = "class_names.json"





def load_class_names(path):
    if not path or not os.path.exists(path):
        print(f"[warn] class names file not found: {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        max_idx = max(int(k) for k in data.keys())
        out = ["<unknown>"] * (max_idx + 1)
        for k, v in data.items():
            out[int(k)] = v
        return out

    print(f"[warn] unsupported class names format: {type(data)}")
    return None








def class_label(class_names, idx):
    if class_names is None:
        return None

    if 0 <= idx < len(class_names):
        return class_names[idx]

    return f"<unknown:{idx}>"








def infer_config_from_state_dict(sd):
    keys = list(sd.keys())

    if any(k.startswith("module.") for k in keys):
        sd = {k[len("module."):]: v for k, v in sd.items()}
        keys = list(sd.keys())

    pos = sd["pos_embed.pos_embed"]
    d_model = pos.shape[-1]
    pos_len = pos.shape[1]

    use_cls_token = "cls_token" in sd
    n_patches = pos_len - (1 if use_cls_token else 0)

    patch_w = sd["patch_embed.proj.weight"]
    in_chans = patch_w.shape[1]
    patch_size = patch_w.shape[-1]

    side = int(round(n_patches ** 0.5))
    img_size = side * patch_size

    n_layers = max(int(k.split(".")[1]) for k in keys if k.startswith("blocks.")) + 1
    n_classes = sd["head.weight"].shape[0]
    d_ff = sd["blocks.0.moe.experts.0.w1.weight"].shape[0]
    num_experts = sd["blocks.0.moe.gate.weight"].shape[0]

    q_out = sd["blocks.0.attn.q_w.weight"].shape[0]
    kv_out = sd["blocks.0.attn.k_w.weight"].shape[0]

    candidates = []

    for qh in range(1, d_model + 1):
        if d_model % qh:
            continue

        d_k = d_model // qh

        if qh * d_k != q_out:
            continue

        if kv_out % d_k:
            continue

        kh = kv_out // d_k

        if kh < 1 or qh % kh:
            continue

        candidates.append((qh, kh))

    if not candidates:
        raise RuntimeError(
            f"Cannot infer q_heads/kv_heads from state_dict "
            f"(d_model={d_model}, q_out={q_out}, kv_out={kv_out})."
        )

    q_heads, kv_heads = candidates[0]

    if len(candidates) > 1:
        print(f"[warn] multiple head configs possible: {candidates}. using {candidates[0]}.")
        print(f"       if output looks wrong, try --q_heads/--kv_heads.")

    cfg = dict(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        n_classes=n_classes,
        d_model=d_model,
        n_layers=n_layers,
        q_heads=q_heads,
        kv_heads=kv_heads,
        d_ff=d_ff,
        num_experts=num_experts,
        use_cls_token=use_cls_token,
    )

    return cfg








def build_model_from_cfg(cfg):
    return ViT(
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        in_chans=cfg["in_chans"],
        n_classes=cfg["n_classes"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        q_heads=cfg["q_heads"],
        kv_heads=cfg["kv_heads"],
        d_ff=cfg["d_ff"],
        num_experts=cfg["num_experts"],
        K=cfg.get("K", 2),
        alpha=cfg.get("alpha", 0.01),
        use_cls_token=cfg["use_cls_token"],
    )








def load_model(ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu", overrides=None):
    ckpt = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    sd = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt

    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    cfg = infer_config_from_state_dict(sd)

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v

    print("[cfg] detected:", cfg)

    model = build_model_from_cfg(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    if missing:
        print(f"[warn] missing keys: {len(missing)} (first 5: {missing[:5]})")

    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

    model.to(device).eval()

    return model, cfg








def build_val_transform(img_size):
    return T.Compose([
        T.Resize(int(img_size * 256 / 224)),
        T.CenterCrop(img_size),
        T.ToTensor(),
    ])








def normalize(x):
    return (x - MEAN) / STD








@torch.no_grad()
def predict_image(model, image, cfg, device="cpu", topk=5, class_names=None):
    tf = build_val_transform(cfg["img_size"])

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")

    x = normalize(tf(image)).unsqueeze(0).to(device)
    logits, _ = model(x)
    probs = F.softmax(logits.float(), dim=-1)[0]
    conf, pred = probs.topk(topk)

    idxs = pred.cpu().tolist()

    result = {
        "top_indices": idxs,
        "top_probs":   conf.cpu().tolist(),
    }

    if class_names is not None:
        result["top_names"] = [class_label(class_names, i) for i in idxs]

    return result








def _parse_args():
    p = argparse.ArgumentParser("ViT-MoE inference (auto-config)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--topk", type=int, default=5)

    p.add_argument(
        "--classes",
        default=CLASS_NAMES_JSON,
        help=f"path to class names JSON (default: {CLASS_NAMES_JSON})",
    )

    p.add_argument("--img_size",    type=int, default=None)
    p.add_argument("--patch_size",  type=int, default=None)
    p.add_argument("--d_model",     type=int, default=None)
    p.add_argument("--n_layers",    type=int, default=None)
    p.add_argument("--n_classes",   type=int, default=None)
    p.add_argument("--q_heads",     type=int, default=None)
    p.add_argument("--kv_heads",    type=int, default=None)
    p.add_argument("--d_ff",        type=int, default=None)
    p.add_argument("--num_experts", type=int, default=None)

    return p.parse_args()








def main():
    args = _parse_args()

    overrides = dict(
        img_size=args.img_size,
        patch_size=args.patch_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_classes=args.n_classes,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        d_ff=args.d_ff,
        num_experts=args.num_experts,
    )

    class_names = load_class_names(args.classes)

    if class_names is not None:
        print(f"[info] loaded {len(class_names)} class names from {args.classes}")

    print(f"[info] device = {args.device}")
    print(f"[info] loading {args.ckpt}")

    model, cfg = load_model(args.ckpt, device=args.device, overrides=overrides)

    if class_names is not None and len(class_names) != cfg["n_classes"]:
        print(
            f"[warn] class names count ({len(class_names)}) "
            f"!= model n_classes ({cfg['n_classes']})."
        )

    if args.image:
        res = predict_image(
            model, args.image, cfg,
            device=args.device,
            topk=args.topk,
            class_names=class_names,
        )

        print(f"\n{args.image}")

        for i, (idx, p) in enumerate(zip(res["top_indices"], res["top_probs"])):
            if "top_names" in res and res["top_names"][i] is not None:
                name = res["top_names"][i]
                print(f"  #{i+1}: {name} | class {idx} | prob {p:.4f}")
            else:
                print(f"  #{i+1}: class {idx:>4d} | prob {p:.4f}")
    else:
        print("[warn] --image is required.")








if __name__ == "__main__":
    main()