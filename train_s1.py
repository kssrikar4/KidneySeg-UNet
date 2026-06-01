import os
import pandas as pd
import numpy as np
import torch
import monai
from monai.transforms import Compose, EnsureChannelFirstd, Resized, RandRotated, RandFlipd, MapTransform
from monai.data import DataLoader
from monai.losses import DiceFocalLoss
from monai.networks.nets import SegResNet

class LoadNpyDictd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.load(d[key])
            if key == "image" and len(d[key].shape) == 3: d[key] = d[key][np.newaxis, ...]
        return d

class S1Labeld(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys: d[key] = (d[key] > 0).astype(np.float32)
        return d

def get_s1_transforms():
    train_tf = Compose([
        LoadNpyDictd(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        S1Labeld(keys=["label"]),
        Resized(keys=["image", "label"], spatial_size=(128, 128, 128), mode=("trilinear", "nearest")),
        RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.5),
    ])
    val_tf = Compose([
        LoadNpyDictd(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        S1Labeld(keys=["label"]),
        Resized(keys=["image", "label"], spatial_size=(128, 128, 128), mode=("trilinear", "nearest")),
    ])
    return train_tf, val_tf

def train_s1():
    df = pd.read_csv("dataset.csv")
    train_data = [{"image": f"preprocessed/{r['PatientID']}/ct.npy", "label": f"preprocessed/{r['PatientID']}/mask.npy"} for _, r in df[df['split'] == 'train'].iterrows()]
    val_data = [{"image": f"preprocessed/{r['PatientID']}/ct.npy", "label": f"preprocessed/{r['PatientID']}/mask.npy"} for _, r in df[df['split'] == 'val'].iterrows()]
    
    train_tf, val_tf = get_s1_transforms()
    train_loader = DataLoader(monai.data.Dataset(train_data, transform=train_tf), batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(monai.data.Dataset(val_data, transform=val_tf), batch_size=1, shuffle=False)
    
    device = torch.device("cuda")
    model = SegResNet(spatial_dims=3, in_channels=2, out_channels=1, init_filters=16).to(device)
    loss_func = DiceFocalLoss(sigmoid=True, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scaler = torch.amp.GradScaler('cuda')
    
    best_dice = -1.0
    for epoch in range(100):
        model.train()
        for batch in train_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = loss_func(model(inputs), labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch["image"].to(device), batch["label"].to(device)
                pred = (model(inputs).sigmoid() > 0.5).float()
                dice = (2. * (pred * labels).sum()) / (pred.sum() + labels.sum() + 1e-8)
                if pred.sum() == 0 and labels.sum() == 0: dice = torch.tensor(1.0)
                val_dice += dice.item()
        val_dice /= len(val_loader)
        
        print(f"Epoch {epoch+1}, Val Dice: {val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_model_s1.pth")
            print("Saved Best S1.")
        scheduler.step()

if __name__ == "__main__":
    train_s1()
