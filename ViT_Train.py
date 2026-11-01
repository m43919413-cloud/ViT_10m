import os
import math
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from timm.data.mixup import Mixup
from ViT_model import ViT
from Transformer_Dataloader import build_imagenet100_loaders





def get_drive_save_dir():

    candidates = [

        "/content/drive/MyDrive/ViT_10M_checkpoints",

        "./ViT_10M_checkpoints",

    
    ]
    for c in candidates:

        parent = os.path.dirname(c)

        if os.path.isdir(parent):

            os.makedirs(c, exist_ok=True)

            return c
        
    
    
    os.makedirs("./ViT_10M_checkpoints", exist_ok=True)

    return "./ViT_10M_checkpoints"


def is_main():

    return (not dist.is_initialized()) or dist.get_rank() == 0


def find_latest_checkpoint(save_dir):
    
    if not save_dir or not os.path.isdir(save_dir):

        return None
    
    ckpts = sorted([

        f for f in os.listdir(save_dir)

        if f.startswith("ckpt_epoch_") and f.endswith(".pt")

    ])

    if not ckpts:

        return None
    
    return os.path.join(save_dir, ckpts[-1])


def _cleanup_old_checkpoints(save_dir, keep_last_n):

    if not keep_last_n or keep_last_n <= 0:

        return
    
    ckpts = sorted([

        f for f in os.listdir(save_dir)

        if f.startswith("ckpt_epoch_") and f.endswith(".pt")

    ])

    for f in ckpts[:-keep_last_n]:

        try:

            os.remove(os.path.join(save_dir, f))

        except OSError:

            pass


        


def train(model, train_loader, val_loader, epochs=200, lr=3e-3,
          device="cuda", aux_weight=0.01, clip=1.0,
          save_path="ViT.pt", use_amp=True, warmup=1000,
          ddp=False, local_rank=0,
          save_every=2, save_dir=None,
          keep_last_n=3,
          early_stop_acc=80.0,
          resume=True):

    if save_dir is None:
        save_dir = get_drive_save_dir()

    model = model.to(device)

    if ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    mixup_fn = Mixup(
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        prob=0.5,
        switch_prob=0.5,
        label_smoothing=0.1,
        num_classes=100,
    )

    try:

        steps_per_epoch = len(train_loader)

    except TypeError:

        steps_per_epoch = None

    total_steps = steps_per_epoch * epochs if steps_per_epoch else None

    def lr_lambda(step):

        if step < warmup:

            return (step + 1) / warmup
        
        if total_steps:

            p = (step - warmup) / max(1, total_steps - warmup)

            return 0.5 * (1 + math.cos(math.pi * p))
        
        return 1.0





    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    best_val = 0.0

    start_epoch = 0

    stop_training = False

    
    if resume:

        resume_path = find_latest_checkpoint(save_dir)

        if resume_path and os.path.exists(resume_path):

            if is_main():

                print(f"[resume] loading checkpoint: {resume_path}")

            ckpt = torch.load(resume_path, map_location=device)

            stop_training = False


            raw_model = model.module if ddp else model
    
            raw_model.load_state_dict(ckpt["model"])

            optimizer.load_state_dict(ckpt["optimizer"])

            scheduler.load_state_dict(ckpt["scheduler"])

            if "scaler" in ckpt and ckpt["scaler"] is not None:

                scaler.load_state_dict(ckpt["scaler"])

            start_epoch = ckpt.get("epoch", 0) + 1

            best_val = ckpt.get("best_val", 0.0)

            if is_main():

                cur_lr = optimizer.param_groups[0]["lr"]

                print(f"[resume] start_epoch={start_epoch} | "
                      f"best_val={best_val:.2f} | "
                      f"LR={cur_lr:.3e} | "
                      f"scheduler.last_epoch={scheduler.last_epoch}")
                
        elif is_main():

            print("[resume] no checkpoint found, training from scratch.")



    for epoch in range(start_epoch, epochs):


        if ddp and hasattr(train_loader.sampler, "set_epoch"):

            train_loader.sampler.set_epoch(epoch)


        model.train()

        total_loss, correct, total = 0.0, 0, 0

        n_batches = 0

        t0 = time.time()


        for x, y in train_loader:

            x = x.to(device, non_blocking=True)

            y = y.to(device, non_blocking=True)

            y_true = y


            if mixup_fn is not None:

                x, y = mixup_fn(x, y)


            optimizer.zero_grad(set_to_none=True)


            with torch.amp.autocast("cuda", enabled=use_amp):

                logits, aux = model(x)

                loss = criterion(logits, y) + aux_weight * aux


            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            scaler.step(optimizer)

            scaler.update()

            scheduler.step()


            total_loss += loss.item()

            correct += (logits.argmax(-1) == y_true).sum().item()

            total += y_true.size(0)

            n_batches += 1



        model.eval()

        val_correct, val_total = 0, 0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(device, non_blocking=True)

                y = y.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):

                    logits, _ = model(x)

                val_correct += (logits.argmax(-1) == y).sum().item()

                val_total += y.size(0)



        if ddp:

            t = torch.tensor(

                [correct, total, val_correct, val_total,
                stop_training == False,

                 total_loss, n_batches],
                 
                device=device, dtype=torch.float64,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            correct, total, val_correct, val_total, total_loss, n_batches = t.tolist()

        val_acc = 100 * val_correct / max(val_total, 1)
        train_acc = 100 * correct / max(total, 1)
        cur_lr = optimizer.param_groups[0]["lr"]

        if is_main():
            print(f"epoch {epoch:02d} | "
                  f"loss {total_loss / max(n_batches, 1):.4f} | "
                  f"train {train_acc:.2f}% | "
                  f"val {val_acc:.2f}% | "
                  f"lr {cur_lr:.2e} | "
                  f"{time.time() - t0:.1f}s")

            raw_model = model.module if ddp else model

            if val_acc > best_val:
                best_val = val_acc
                torch.save(raw_model.state_dict(), save_path)
                print(f"   saved best ({val_acc:.2f}%)")

            if (epoch + 1) % save_every == 0:
                ckpt_path = os.path.join(
                    save_dir, f"ckpt_epoch_{epoch + 1:03d}.pt"
                )
                torch.save({
                    "epoch": epoch,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val": best_val,
                }, ckpt_path)
                _cleanup_old_checkpoints(save_dir, keep_last_n)
                print(f"   checkpoint saved: {ckpt_path}")

            if val_acc >= early_stop_acc:
                print(f"   early stop: val_acc {val_acc:.2f}% >= {early_stop_acc}%")
                stop_training = True

        if ddp:
            stop_t = torch.tensor([1 if stop_training else 0], device=device)
            dist.broadcast(stop_t, src=0)
            stop_training = bool(stop_t.item())

        if stop_training:
            break

    return model


def make_model():

    return ViT(
        img_size=224,
        patch_size=16,
        in_chans=3,
        n_classes=100,
        d_model=224,
        n_layers=8,
        q_heads=8,
        kv_heads=2,
        d_ff=608,
        num_experts=4,
        K=2,
    )


def ddp_worker(local_rank, world_size, cfg):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(
        backend="nccl",
        rank=local_rank,
        world_size=world_size,
    )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    train_loader, val_loader, n_tr, n_va = build_imagenet100_loaders(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    if is_main():
        print(f"train={n_tr} | val={n_va} | world_size={world_size}")

    model = make_model()

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        device=device,
        save_path=cfg["save_path"],
        use_amp=True,
        ddp=True,
        local_rank=local_rank,
        save_every=cfg["save_every"],
        keep_last_n=cfg["keep_last_n"],
        early_stop_acc=cfg["early_stop_acc"],
        resume=cfg["resume"],
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print(f"detected GPUs: {n_gpus}")

    cfg = dict(
        db_path="/content/drive/MyDrive/imagenet.db",
        batch_size=256,
        num_workers=4,
        epochs=200,
        lr=3e-4,
        save_path="/content/drive/MyDrive/ViT_10M_checkpoints/ViT_best.pt",
        save_every=2,
        keep_last_n=3,
        early_stop_acc=90.0,
        resume=True, 
    )

    if n_gpus >= 2:
        print(f"launching DDP on {n_gpus} GPUs...")
        mp.spawn(
            ddp_worker,
            args=(n_gpus, cfg),
            nprocs=n_gpus,
            join=True,
        )
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device:", device)

        model = make_model()
        train_loader, val_loader, n_tr, n_va = build_imagenet100_loaders(
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
        )
        print(f"train={n_tr} | val={n_va}")

        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            device=device,
            save_path=cfg["save_path"],
            use_amp=True,
            ddp=False,
            save_every=cfg["save_every"],
            keep_last_n=cfg["keep_last_n"],
            early_stop_acc=cfg["early_stop_acc"],
            resume=cfg["resume"],
        )