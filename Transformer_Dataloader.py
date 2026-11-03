import io
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T
from torchvision.transforms import RandAugment
from datasets import load_dataset




MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class ImageNet100Arrow(Dataset):

    def __init__(self, hf_split, img_size=224, augment=False, normalize=True):

        super().__init__()

        self.ds = hf_split                

        self.normalize = normalize


        if augment:

            self.tf = T.Compose([

                T.RandomResizedCrop(img_size, scale=(0.35, 1.0), ratio=(3/4, 4/3)),

                T.RandomHorizontalFlip(),

                RandAugment(num_ops=2, magnitude=7),

                T.ToTensor(),

                T.RandomErasing(p=0.25, scale=(0.02, 0.2)),
            ])

        else:

            self.tf = T.Compose([

                T.Resize(int(img_size * 256 / 224)),

                T.CenterCrop(img_size),

                T.ToTensor(),

            ])


    def __len__(self):

        return len(self.ds)


    def __getitem__(self, idx):

        row = self.ds[idx]

        img = row["image"]

        
        if isinstance(img, Image.Image):

            if img.mode != "RGB":

                img = img.convert("RGB")

        else:

            img = Image.open(io.BytesIO(img)).convert("RGB")


        x = self.tf(img)

        if self.normalize:

            x = (x - MEAN) / STD

        return x, int(row["label"])




def build_imagenet100_loaders(repo_id="clane9/imagenet-100",
                              
                              batch_size=128, num_workers=4,

                              eval_batch_size=None, img_size=224):
    
    eval_bs = eval_batch_size or batch_size * 2


    
    data = load_dataset(repo_id, split=None)
    
    train_hf = data["train"]

    val_hf   = data["validation"]


    train_set = ImageNet100Arrow(train_hf, img_size=img_size, augment=True)

    val_set   = ImageNet100Arrow(val_hf,   img_size=img_size, augment=False)


    ddp = dist.is_available() and dist.is_initialized()

    train_sampler = DistributedSampler(train_hf, shuffle=True,  drop_last=True)  if ddp else None

    val_sampler   = DistributedSampler(val_hf,   shuffle=False, drop_last=False) if ddp else None


    train_loader = DataLoader(

        train_set,

        batch_size=batch_size,

        shuffle=(train_sampler is None),

        sampler=train_sampler,

    
        num_workers=num_workers,

        pin_memory=True,

        persistent_workers=(num_workers > 0),

        prefetch_factor=4 if num_workers > 0 else None,

        drop_last=True,

    )

    val_loader = DataLoader(

        val_set,

        batch_size=eval_bs,

        shuffle=False,

        sampler=val_sampler,

        num_workers=max(1, num_workers // 2),

        pin_memory=True,

        persistent_workers=(num_workers > 0),

        prefetch_factor=2 if num_workers > 0 else None,

        drop_last=False,

    )


    print(f"[loader] train={len(train_set)} | val={len(val_set)}")

    return train_loader, val_loader, len(train_set), len(val_set)



if __name__ == "__main__":

    tr, va, n_tr, n_va = build_imagenet100_loaders(batch_size=8, num_workers=2)

    print(f"train={n_tr} | val={n_va}")

    for i, (x, y) in enumerate(tr):

        print("batch:", tuple(x.shape), x.dtype,
              
              f"range=[{x.min():.2f},{x.max():.2f}]",

              "labels:", y.min().item(), "-", y.max().item())
        
        if i == 2:
            
            break