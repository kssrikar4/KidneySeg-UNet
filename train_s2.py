import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import monai
from tqdm import tqdm
from monai.transforms import (
    Compose, EnsureTyped, SpatialPadd, RandCropByPosNegLabeld,
    RandRotated, RandFlipd, Resize, RandGaussianSmoothd, 
    RandScaleIntensityd, RandShiftIntensityd, RandGaussianNoised
)
from monai.networks.nets import SegResNet
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from scipy.ndimage import label

set_determinism(seed=42)

def get_bounding_box(mask_arr, margin=15):
    nz = np.any(mask_arr, axis=(1, 2))
    ny = np.any(mask_arr, axis=(0, 2))
    nx = np.any(mask_arr, axis=(0, 1))
    if not np.any(nz):
        z, y, x = mask_arr.shape
        return slice(z//2-48, z//2+48), slice(y//2-64, y//2+64), slice(x//2-64, x//2+64)
    zmin, zmax = np.where(nz)[0][[0, -1]]
    ymin, ymax = np.where(ny)[0][[0, -1]]
    xmin, xmax = np.where(nx)[0][[0, -1]]
    zmin = max(0, zmin - margin); zmax = min(mask_arr.shape[0], zmax + margin + 1)
    ymin = max(0, ymin - margin); ymax = min(mask_arr.shape[1], ymax + margin + 1)
    xmin = max(0, xmin - margin); xmax = min(mask_arr.shape[2], xmax + margin + 1)
    return slice(zmin, zmax), slice(ymin, ymax), slice(xmin, xmax)

def get_kidney_samples(data_dicts, model_s1, device, train=True):
    samples = []
    resizer_s1 = Resize(spatial_size=(128, 128, 128), mode="trilinear")
    cache_dir = f"cache_s2_{'train' if train else 'val'}"
    if os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0:
        for f in sorted(os.listdir(cache_dir)):
            if f.endswith(".pth"): samples.append(torch.load(os.path.join(cache_dir, f), weights_only=False))
        return samples

    os.makedirs(cache_dir, exist_ok=True)
    for idx, item in enumerate(tqdm(data_dicts)):
        ct_arr, mask_arr = np.load(item["image"]), np.load(item["label"])
        ct_s1 = resizer_s1(torch.tensor(ct_arr).float()).unsqueeze(0).to(device)
        with torch.no_grad():
            out_s1 = (model_s1(ct_s1).sigmoid() > 0.5).float().squeeze(0).cpu().numpy()
        s1_mask_full = Resize(spatial_size=ct_arr.shape[1:], mode="nearest")(torch.tensor(out_s1)).squeeze(0).numpy()
        if not np.any(s1_mask_full) and train: s1_mask_full = (mask_arr > 0).astype(np.float32)
        if not np.any(s1_mask_full): continue

        labeled_mask, num_features = label(s1_mask_full)
        for i in range(1, num_features + 1):
            comp = (labeled_mask == i).astype(np.float32)
            z, y, x = get_bounding_box(comp, margin=15)
            sample = {"image": np.concatenate([ct_arr[:, z, y, x], s1_mask_full[z, y, x][np.newaxis, ...]], axis=0),
                      "label": (mask_arr[z, y, x] == 2).astype(np.float32)[np.newaxis, ...],
                      "patient_id": os.path.basename(os.path.dirname(item["image"]))}
            samples.append(sample); torch.save(sample, os.path.join(cache_dir, f"sample_{idx}_{i}.pth"))
    return samples

def train_s2():
    df = pd.read_csv("dataset.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m1 = SegResNet(spatial_dims=3, in_channels=2, out_channels=1, init_filters=16).to(device)
    if os.path.exists("best_model_s1.pth"): m1.load_state_dict(torch.load("best_model_s1.pth", map_location=device, weights_only=False))
    m1.eval()

    train_data = [{"image": f"preprocessed/{r['PatientID']}/ct.npy", "label": f"preprocessed/{r['PatientID']}/mask.npy"} for _, r in df[df['split'] == 'train'].iterrows()]
    val_data = [{"image": f"preprocessed/{r['PatientID']}/ct.npy", "label": f"preprocessed/{r['PatientID']}/mask.npy"} for _, r in df[df['split'] == 'val'].iterrows()]
    
    train_samples = get_kidney_samples(train_data, m1, device, train=True)
    val_samples = get_kidney_samples(val_data, m1, device, train=False)
    del m1; torch.cuda.empty_cache()
    
    roi = (128, 128, 128)
    train_tf = Compose([
        EnsureTyped(keys=["image", "label"]), SpatialPadd(keys=["image", "label"], spatial_size=roi),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=roi, pos=3, neg=1, num_samples=2),
        RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3, prob=0.4),
        RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
        RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15), prob=0.1),
        RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.15),
    ])
    
    train_loader = DataLoader(Dataset(data=train_samples, transform=train_tf), batch_size=1, shuffle=True, num_workers=4)
    model = SegResNet(spatial_dims=3, in_channels=3, out_channels=1, init_filters=32, blocks_down=[1, 2, 2, 4]).to(device)
    loss_func = monai.losses.DiceFocalLoss(sigmoid=True, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    scaler = torch.cuda.amp.GradScaler(); dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    best_dice = -1.0
    for epoch in range(150):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = loss_func(model(inputs), labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        
        model.eval(); dice_metric.reset()
        with torch.no_grad():
            for s in val_samples:
                img = torch.tensor(s["image"]).float().unsqueeze(0).to(device)
                lbl = torch.tensor(s["label"]).float().unsqueeze(0).to(device)
                with torch.cuda.amp.autocast():
                    out = sliding_window_inference(img, roi_size=roi, sw_batch_size=4, predictor=model)
                dice_metric(y_pred=(out.sigmoid() > 0.5).float(), y=lbl)

        avg_dice = dice_metric.aggregate().item()
        print(f"Epoch {epoch+1} Dice: {avg_dice:.4f}")
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), "best_model_s2.pth")
            print("Saved Best S2.")
        scheduler.step()

if __name__ == "__main__":
    train_s2()
