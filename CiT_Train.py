import os
import time
import shutil
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed
from deepspeed.utils import logger
from CiT import CiT
from timm.data.mixup import Mixup
import json
from Transformer_Dataloader import build_imagenet100_loaders







def is_main_rank() -> bool:

    return (not dist.is_initialized()) or dist.get_rank() == 0


def pick_save_dir() -> str:

    for c in ("/content/drive/MyDrive/CiT_checkpoints", "./CiT_checkpoints"):

        try:

            os.makedirs(c, exist_ok=True)

            return c
        
        except OSError:

            continue
        
    os.makedirs("./CiT_checkpoints", exist_ok=True)

    return "./CiT_checkpoints"

    




def find_latest_tag(ckpt_dir: str):

    if not os.path.isdir(ckpt_dir):

        return None
    
    tags = [d for d in os.listdir(ckpt_dir)
            
            if d.startswith("epoch_") and os.path.isdir(os.path.join(ckpt_dir, d))]
    
    if not tags:

        return None
    
    tags.sort(key=lambda s: int(s.split("_")[1]))

    return tags[-1]





def cleanup_old_tags(ckpt_dir: str, keep_last_n: int):

    if keep_last_n <= 0:

        return
    
    tags = [d for d in os.listdir(ckpt_dir)
            
            if d.startswith("epoch_") and os.path.isdir(os.path.join(ckpt_dir, d))]
    
    tags.sort(key=lambda s: int(s.split("_")[1]))

    for old in tags[:-keep_last_n]:

        shutil.rmtree(os.path.join(ckpt_dir, old), ignore_errors=True)







def train_one_epoch(engine, loader, mixup_fn, aux_weight, epoch, device):

    engine.train()

    if hasattr(loader.sampler, "set_epoch"):

        loader.sampler.set_epoch(epoch)

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    n_batches = 0

    t0 = time.time()





    for x, y in loader:

        x = x.to(device, non_blocking=True)

        y = y.to(device, non_blocking=True)

        y_hard = y

        if mixup_fn is not None:

            x, y = mixup_fn(x, y)

            y_hard = y.argmax(-1) if y.dim() > 1 else y


        logits, aux = engine(x)

        loss = F.cross_entropy(logits, y) + aux_weight * aux


        engine.backward(loss)

        engine.step()

        if (n_batches % 50) == 0:

            torch.cuda.empty_cache()

        with torch.no_grad():

            total_loss += loss.detach().item()

            total_correct += (logits.argmax(-1) == y_hard).sum().item()

            total_samples += y_hard.size(0)

            n_batches += 1




    stats = torch.tensor(

        [total_loss, total_correct, total_samples, n_batches],

        device=device, dtype=torch.float16,

    )

    if dist.is_initialized():

        dist.all_reduce(stats, op=dist.ReduceOp.SUM)



    mean_loss = stats[0].item() / max(stats[3].item(), 1)

    acc = 100.0 * stats[1].item() / max(stats[2].item(), 1)

    return mean_loss, acc, time.time() - t0





@torch.no_grad()

def evaluate(engine, loader, device):

    engine.eval()

    correct, total = 0, 0

    for x, y in loader:

        x = x.to(device, non_blocking=True)

        y = y.to(device, non_blocking=True)

        logits, _ = engine(x)

        correct += (logits.argmax(-1) == y).sum().item()

        total += y.size(0)



    stats = torch.tensor([correct, total], device=device, dtype=torch.float16)

    if dist.is_initialized():

        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    return 100.0 * stats[0].item() / max(stats[1].item(), 1)









def save_ckpt(engine, ckpt_dir, tag, client_state):

    engine.save_checkpoint(ckpt_dir, tag=tag, client_state=client_state)




def train_engine(engine, train_loader, val_loader, args, device):

    mixup_fn = Mixup(

        mixup_alpha=0.2,

        cutmix_alpha=1.0,

        prob=0.5,

        switch_prob=0.5,

        label_smoothing=0.1,

        num_classes=args.num_classes,

    )


    start_epoch = 0

    best_val = 0.0

    
    if args.resume:
        
        latest = find_latest_tag(args.ckpt_dir)

        if latest is not None:

            if is_main_rank():

                print(f"[resume] loading {latest}")

            _, client_state = engine.load_checkpoint(args.ckpt_dir, tag=latest)

            if client_state:

                start_epoch = client_state.get("epoch", 0) + 1

                best_val = client_state.get("best_val", 0.0)

                if is_main_rank():

                    logger.info(f"[resume] start_epoch={start_epoch} best_val={best_val:.2f}")

        elif is_main_rank():

            logger.info("[resume] no checkpoint found, training from scratch.")


    
    for epoch in range(start_epoch, args.epochs):

        tr_loss, tr_acc, dt = train_one_epoch(

            engine, train_loader, mixup_fn, args.aux_weight, epoch, device)
        
        val_acc = evaluate(engine, val_loader, device)

        cur_lr = engine.get_lr()[0] if hasattr(engine, "get_lr") else args.lr

        if is_main_rank():

            print(

                f"epoch {epoch:02d} | loss {tr_loss:.4f} | "

                f"train {tr_acc:.2f}% | val {val_acc:.2f}% | "

                f"lr {cur_lr:.2e} | {dt:.1f}s"
            )




        if val_acc > best_val:

            best_val = val_acc

            if is_main_rank():
                
                best_path = os.path.join(args.ckpt_dir, "best")

                os.makedirs(best_path, exist_ok=True)

                engine.save_16bit_model(best_path, save_filename="model.pt")

                print(f"   saved best ({best_val:.2f}%)")





    
        if (epoch + 1) % args.save_every == 0:

            tag = f"epoch_{epoch + 1:03d}"

            client_state = {

                "epoch": epoch,

                "best_val": best_val,
            }

            save_ckpt(engine, args.ckpt_dir, tag, client_state)

            cleanup_old_tags(args.ckpt_dir, args.keep_last_n)

            if is_main_rank():

                logger.info(f"   checkpoint saved: {tag}")


        
        stop = val_acc >= args.early_stop_acc

        stop_t = torch.tensor([1 if stop else 0], device=device)

        if dist.is_initialized():

            dist.broadcast(stop_t, src=0)

        if stop_t.item():

            if is_main_rank():
                
                logger.info(f"   early stop (val={val_acc:.2f}%)")

            break
        

    return engine









def build_loaders(batch_size, num_workers, world_size, rank):

    cfg = dict(
            db_path="/content/drive/MyDrive/imagenet.db",
            batch_size=128,
            num_workers=4,
            epochs=200,
            lr= 2.10468228006034e-4,
            save_path="/content/drive/MyDrive/ViT_10M_checkpoints/ViT_best.pt",
            save_every=2,
            keep_last_n=3,
            early_stop_acc=90.0,
            resume=True, 
        )

    
    train_obj, val_obj, _, _ = build_imagenet100_loaders(

        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"]
    )


def build_loaders(batch_size, num_workers, world_size, rank):
    cfg = dict(
        db_path="/content/drive/MyDrive/imagenet.db",
        batch_size=128,
        num_workers=4,
        epochs=200,
        lr= 2.10468228006034e-4,
        save_path="/content/drive/MyDrive/ViT_10M_checkpoints/ViT_best.pt",
        save_every=2,
        keep_last_n=3,
        early_stop_acc=90.0,
        resume=True,
    )

    train_obj, val_obj, _, _ = build_imagenet100_loaders(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )


  
    train_ds =  train_ds = train_obj.dataset if isinstance(train_obj, DataLoader) else train_obj
    
    val_ds = val_obj.dataset if isinstance(val_obj, DataLoader) else val_obj



    train_sampler = DistributedSampler(

        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,

    )

    val_sampler = DistributedSampler(

        val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False,
    )


    train_loader = DataLoader(

        train_ds, batch_size=batch_size, sampler=train_sampler,

        num_workers=num_workers, pin_memory=True, drop_last=True,

        persistent_workers=num_workers > 0,

    )

    val_loader = DataLoader(

        val_ds, batch_size=batch_size, sampler=val_sampler,

        num_workers=num_workers, pin_memory=True, drop_last=False,

        persistent_workers=num_workers > 0,

    )


    return train_loader, val_loader, len(train_ds), len(val_ds)








def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--epochs", type=int, default=200)

    p.add_argument("--lr", type=float, default=3e-4)

    p.add_argument("--aux_weight", type=float, default=0.01)

    p.add_argument("--num_classes", type=int, default=100)

    p.add_argument("--batch_size", type=int, default=64)

    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--save_every", type=int, default=2)

    p.add_argument("--keep_last_n", type=int, default=3)

    p.add_argument("--early_stop_acc", type=float, default=90.0)

    p.add_argument("--resume", action="store_true")

    p.add_argument("--ckpt_dir", type=str, default=None)

    p.add_argument("--local_rank", type=int, default=-1)

    p = deepspeed.add_config_arguments(p)

    return p.parse_args()





def main():


    #error in this function

    args = parse_args()

    if args.ckpt_dir is None:

        args.ckpt_dir = pick_save_dir()



    cfg = dict(
            db_path="/content/drive/MyDrive/imagenet.db",
            batch_size=128,
            num_workers=4,
            epochs=200,
            lr= 2.10468228006034e-4,
            save_path="/content/drive/MyDrive/ViT_10M_checkpoints/ViT_best.pt",
            save_every=2,
            keep_last_n=3,
            early_stop_acc=90.0,
            resume=True, 
        )

    


    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    if local_rank < 0:

        local_rank = 0

    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")




    
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    rank = int(os.environ.get("RANK", 0))


    train_loader, val_loader, n_tr, n_va = build_loaders(

        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        rank=rank,
        world_size=world_size
    )

    if is_main_rank():

        print(f"train={n_tr} | val={n_va} | world_size={world_size}")




    model = CiT(
        img_size=224, patch_size=16, in_chans=3,
        num_classes=args.num_classes,
        d_model=224, n_layers=8, q_heads=8, kv_heads=2, d_ff=608,
        num_experts=4, K=2, alpha=0.01,
        in_channels=3, kernel_size=3, stride=1, padding=1, dropout=0.1,
    )

       

    with open(args.deepspeed_config) as f:

        ds_config = json.load(f)

    ds_config["optimizer"]["params"]["lr"] = args.lr

    args.deepspeed_config = None



    engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=None,
        config=ds_config,
    )


    train_engine(engine, train_loader, val_loader, args, device)


if __name__ == "__main__":

    main()
   