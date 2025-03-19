#!/usr/bin/env python

print("In Python")

# replace print so it always flushes
def print(*args, **kwargs):
    __builtins__.print(*args, **kwargs, flush=True)

print("Importing libraries...")

import glob
import os
import json
import time
import datetime
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import torchio as tio
import scipy.ndimage
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from skimage.measure import marching_cubes, mesh_surface_area

# --- Command-line arguments and settings ---
print(f"=== {sys.argv[1]}-{sys.argv[2]} ===")
model_name = sys.argv[1]
fold_id = int(sys.argv[2])
random_state = 42
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S')
model_instance = f"{model_name}-{timestamp}-{fold_id}"
print(f"Model instance: {model_instance}")

batch_size = 6
training_epochs = 4000
lr = 0.003
ce_loss_weights = torch.Tensor([1, 1, 1])

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"GPU is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU is NOT available. Using CPU.")

# --- File organization and dataset setup ---
bids_dir = "bids"
session_dirs = []
for json_path in sorted(glob.glob(os.path.join(bids_dir, "sub*", "ses*", "anat", "*echo-01*mag*json"))):
    with open(json_path, 'r') as jf:
        json_data = json.load(jf)
        if json_data['ProtocolName'] in ["wip_iSWI_fl3d_vibe_TRY THIS ONE"]:
            session_dirs.append(os.sep.join(os.path.split(json_path)[0].split(os.sep)[:-1]))
print(f"{len(session_dirs)} sessions found.")

session_dirs = sorted(set([s for s in session_dirs if 'sub-z0449294' not in s]))
print(f"{len(session_dirs)} sessions found.")

extra_files = sum((glob.glob(os.path.join(session_dir, "extra_data", "*.nii*"))
                   for session_dir in session_dirs), [])

qsm_files2 = sorted(sum((glob.glob(os.path.join(session_dir, "extra_data", "*real.nii*"))
                         for session_dir in session_dirs), []))
qsm_files = [x for x in sorted(glob.glob(os.path.join("out/qsm/*.nii")))
             if 'sub-z0449294' not in x]
t2s_files = [x for x in sorted(glob.glob(os.path.join("out/t2s/*.nii")))
             if 'sub-z0449294' not in x]
r2s_files = [x for x in sorted(glob.glob(os.path.join("out/r2s/*.nii")))
             if 'sub-z0449294' not in x]
swi_files = [x for x in sorted(glob.glob(os.path.join("out/swi/*swi.nii")))
             if 'sub-z0449294' not in x]
mag_files = [x for x in sorted(sum((glob.glob(os.path.join(session_dir, "extra_data", "magnitude_combined.nii"))
                                   for session_dir in session_dirs), []))
             if 'sub-z0449294' not in x]
fmap_files = sorted([ef for ef in extra_files if 'B0.nii' in ef])

gre_seg_files = sorted([ef for ef in extra_files if all(p in ef for p in ['segmentation_clean.', 'tgv'])])
t1_files = sorted([ef for ef in extra_files if 'T1w_resliced' in ef and 'homogeneity-corrected' in ef])

ct_files = sorted([ef for ef in extra_files if 'resliced.nii' in ef and 'T1w' not in ef and 'segmentation' not in ef])
ct_seg_files = sorted([ct.replace(".nii", "_segmentation.nii")
                        for ct in ct_files if os.path.exists(ct.replace(".nii", "_segmentation.nii"))])
ct_seg_clean_files = sorted([ct.replace(".nii", "_clean.nii")
                              for ct in ct_seg_files if os.path.exists(ct.replace(".nii", "_clean.nii"))])

print(f"{len(fmap_files)} field maps found.")
print(f"{len(ct_files)} CT images found.")
print(f"{len(ct_seg_files)} raw CT segmentations found.")
print(f"{len(ct_seg_clean_files)} clean CT segmentations found.")
print(f"{len(gre_seg_files)} clean GRE segmentations found.")
print(f"{len(qsm_files)} QSM images found.")
print(f"{len(qsm_files2)} QSM (2) images found.")
print(f"{len(mag_files)} magnitude images found.")
print(f"{len(t2s_files)} T2* maps found.")
print(f"{len(r2s_files)} R2* maps found.")
print(f"{len(swi_files)} SWI maps found.")
print(f"{len(t1_files)} T1w files found.")

assert len(qsm_files) == len(gre_seg_files)
assert len(qsm_files) == len(t2s_files)
assert len(qsm_files) == len(r2s_files)
assert len(qsm_files) == len(swi_files)
assert len(qsm_files) == len(mag_files)
assert len(qsm_files) == len(t1_files)
assert len(ct_files) == len(ct_seg_clean_files)

model_data = { 
    'CT'         : { 'ct_files': ct_files,         'seg_files': ct_seg_clean_files },
    'QSM'        : { 'qsm_files': qsm_files,       'seg_files': gre_seg_files },
    'QSM-SWI'    : { 'qsm_files': qsm_files,       'swi_files': swi_files,      'seg_files': gre_seg_files },
    'QSM-T1-R2s' : { 'qsm_files': qsm_files,       't1_files': t1_files,        'r2s_files': r2s_files, 'seg_files': gre_seg_files },
    'QSM-T1'     : { 'qsm_files': qsm_files,       't1_files': t1_files,        'seg_files': gre_seg_files },
    'T1'         : { 't1_files': t1_files,         'seg_files': gre_seg_files },
    'SWI'        : { 'swi_files': swi_files,       'seg_files': gre_seg_files },
    'R2s'        : { 'r2s_files': r2s_files,       'seg_files': gre_seg_files },
    'GRE'        : { 'mag_files': mag_files,       'seg_files': gre_seg_files },
    'QSM-FMAP'   : { 'qsm_files': qsm_files,       'fmap_files': fmap_files,   'seg_files': gre_seg_files },
    'FMAP'       : { 'fmap_files': fmap_files,       'seg_files': gre_seg_files }
}

colors = {
    'CT'         : '#543005',
    'QSM'        : '#8c510a',
    'QSM-SWI'    : '#bf812d',
    'QSM-T1-R2s' : '#dfc27d',
    'QSM-T1'     : '#f6e8c3',
    'T1'         : '#c7eae5',
    'SWI'        : '#80cdc1',
    'R2s'        : '#35978f',
    'GRE'        : '#01665e',
    'QSM-FMAP'   : '#003c30',
    'FMAP'       : '#003c30'
}

df = pd.DataFrame(model_data[model_name])
infile_cols = [k for k in list(model_data[model_name].keys()) if k != 'seg_files']
n_input_channels = len(infile_cols)
print(f"infile_cols: {infile_cols}; n_input_channels: {n_input_channels}")

k_folds = len(qsm_files)
kf = KFold(n_splits=k_folds, random_state=random_state, shuffle=True)
(train_index, valid_index) = list(kf.split(df))[fold_id]

# --- Custom transform: Random Intensity Shift using Gamma Correction ---
class RandomIntensityShift(tio.Transform):
    def __init__(self, gamma_range=(0.8, 1.2), p=0.5):
        super().__init__()
        self.gamma_range = gamma_range
        self.p = p

    def apply_transform(self, subject):
        if torch.rand(1).item() > self.p:
            return subject
        for image_name in subject.keys():
            if isinstance(subject[image_name], tio.ScalarImage):
                img_tensor = subject[image_name].data
                if img_tensor.ndim == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                gamma = np.random.uniform(*self.gamma_range)
                img_tensor = img_tensor.pow(gamma)
                subject[image_name].set_data(img_tensor)
        return subject

# --- Custom transform: Merge Input Channels ---
class MergeInputChannels(tio.Transform):
    def __init__(self, infile_cols):
        super().__init__()
        self.infile_cols = infile_cols

    def apply_transform(self, subject):
        input_tensors = []
        for col in self.infile_cols:
            if col not in subject:
                continue
            tensor = subject[col].data
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            input_tensors.append(tensor)
        if not input_tensors:
            return subject
        merged = torch.cat(input_tensors, dim=0)
        subject['image'] = tio.ScalarImage(tensor=merged, affine=subject[self.infile_cols[0]].affine)
        subject['mask'] = subject['seg']
        return subject

# --- Build training transforms ---
if model_name == "T1":
    training_transforms = tio.Compose([
        RandomIntensityShift(gamma_range=(0.8, 1.2), p=0.75),
        tio.CropOrPad((80, 80, 80)),
        tio.RandomFlip(axes=(0, 1)),
        tio.RandomAffine(degrees=(90, 90, 90)),
        tio.ZNormalization(),
        MergeInputChannels(infile_cols)
    ])
else:
    training_transforms = tio.Compose([
        tio.CropOrPad((80, 80, 80)),
        tio.RandomFlip(axes=(0, 1)),
        tio.RandomAffine(degrees=(90, 90, 90)),
        tio.ZNormalization(),
        MergeInputChannels(infile_cols)
    ])

if 'QSM' in model_name:
    training_transforms.transforms.append(tio.Lambda(lambda subject: subject))

# --- Build TorchIO subjects ---
subjects = []
for _, row in df.iterrows():
    subject_dict = {}
    for col in infile_cols:
        subject_dict[col] = tio.ScalarImage(row[col])
    subject_dict['seg'] = tio.LabelMap(row['seg_files'])
    subject = tio.Subject(**subject_dict)
    subjects.append(subject)

subjects_train = [subjects[i] for i in train_index]
subjects_valid = [subjects[i] for i in valid_index]

train_subjects_dataset = tio.SubjectsDataset(subjects_train, transform=training_transforms)
valid_subjects_dataset = tio.SubjectsDataset(subjects_valid, transform=training_transforms)

class TorchIODatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subjects_dataset):
        self.subjects_dataset = subjects_dataset
    def __len__(self):
        return len(self.subjects_dataset)
    def __getitem__(self, index):
        subject = self.subjects_dataset[index]
        if 'image' not in subject:
            raise KeyError("Missing 'image' key in subject")
        image = subject['image'].data  # (C, D, H, W)
        if 'mask' not in subject:
            raise KeyError("Missing 'mask' key in subject")
        mask = subject['mask'].data.squeeze(0)  # (D, H, W)
        mask = mask.long()  # Ensure mask is Long for cross_entropy
        return image, mask

train_dataset = TorchIODatasetWrapper(train_subjects_dataset)
valid_dataset = TorchIODatasetWrapper(valid_subjects_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# --- 3D UNet model definition ---
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, channels=(16, 32, 64, 128, 256)):
        super(UNet3D, self).__init__()
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        prev_channels = in_channels
        for ch in channels:
            self.encoders.append(self.conv_block(prev_channels, ch))
            prev_channels = ch
        self.bottleneck = self.conv_block(prev_channels, prev_channels * 2)
        rev_channels = list(reversed(channels))
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        cur_channels = prev_channels * 2
        for ch in rev_channels:
            self.upconvs.append(nn.ConvTranspose3d(cur_channels, ch, kernel_size=2, stride=2))
            self.decoders.append(self.conv_block(ch * 2, ch))
            cur_channels = ch
        self.final_conv = nn.Conv3d(cur_channels, out_channels, kernel_size=1)
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        enc_feats = []
        for encoder in self.encoders:
            x = encoder(x)
            enc_feats.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upconv, decoder, enc in zip(self.upconvs, self.decoders, reversed(enc_feats)):
            x = upconv(x)
            if x.shape != enc.shape:
                diffZ = enc.size()[2] - x.size()[2]
                diffY = enc.size()[3] - x.size()[3]
                diffX = enc.size()[4] - x.size()[4]
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2,
                              diffZ // 2, diffZ - diffZ // 2])
            x = torch.cat([enc, x], dim=1)
            x = decoder(x)
        return self.final_conv(x)

model = UNet3D(in_channels=n_input_channels, out_channels=3).to(device)

# --- Loss functions (CE + Dice) ---
ce_loss_fn = nn.CrossEntropyLoss(weight=ce_loss_weights.to(device))
def dice_loss(pred, target, eps=1e-5):
    pred = F.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
    intersection = (pred * target_onehot).sum(dim=(2,3,4))
    sums = pred.sum(dim=(2,3,4)) + target_onehot.sum(dim=(2,3,4))
    dice = (2 * intersection + eps) / (sums + eps)
    return 1 - dice.mean()
def combined_loss(pred, target):
    return ce_loss_fn(pred, target) + dice_loss(pred, target)

# --- Prior loss function: count-aware, size, and sphericity-based roundness ---
def compute_prior_losses(outputs):
    # Use argmax to obtain the final segmentation (assumes class 1 is the marker)
    segmentation = torch.argmax(outputs, dim=1).detach().cpu().numpy()
    B = segmentation.shape[0]
    batch_count_loss = 0
    batch_size_loss = 0
    batch_round_loss = 0
    for i in range(B):
        # Binary mask where class 1 is marked as foreground
        binary_mask = (segmentation[i] == 1).astype(np.uint8)
        labeled, num_components = scipy.ndimage.label(binary_mask)
        count_loss = (num_components - 3)**2
        size_loss = 0
        round_loss = 0
        for comp in range(1, num_components + 1):
            comp_mask = (labeled == comp).astype(np.uint8)
            vol = comp_mask.sum()
            # Size penalty: zero if within [15,80]
            if vol < 15:
                size_penalty = (15 - vol)**2
            elif vol > 80:
                size_penalty = (vol - 80)**2
            else:
                size_penalty = 0
            size_loss += size_penalty
            # Compute surface area using marching cubes if possible
            try:
                verts, faces, normals, values = marching_cubes(comp_mask.astype(np.float32), level=0.5)
                area = mesh_surface_area(verts, faces)
                sphericity = (np.pi**(1/3) * (6 * vol)**(2/3)) / area
                round_penalty = (1 - sphericity)**2
            except Exception as e:
                print(f"Marching cubes failed for component {comp} with error: {e}")
                round_penalty = 1.0
            round_loss += round_penalty
        if num_components > 0:
            size_loss /= num_components
            round_loss /= num_components
        batch_count_loss += count_loss
        batch_size_loss += size_loss
        batch_round_loss += round_loss
    batch_count_loss /= B
    batch_size_loss /= B
    batch_round_loss /= B
    return batch_count_loss, batch_size_loss, batch_round_loss

# --- Metric helper functions using connected components ---
def process_volume(pred_vol, targ_vol, calc_vol, structure):
    pred_labels, pred_nlabels = scipy.ndimage.label(pred_vol, structure=structure)
    _, targ_nlabels = scipy.ndimage.label(targ_vol, structure=structure)
    if pred_nlabels > 3:
        sizes = np.bincount(pred_labels.ravel())[1:]
        largest_labels = np.argsort(sizes)[-3:] + 1
        pred_vol = np.isin(pred_labels, largest_labels).astype(pred_vol.dtype)
        pred_labels, pred_nlabels = scipy.ndimage.label(pred_vol, structure=structure)
    overlap = np.logical_and(pred_vol == targ_vol, pred_vol == 1)
    _, n_overlaps = scipy.ndimage.label(overlap, structure=structure)
    labeled_pred, num_pred_objects = scipy.ndimage.label(pred_vol == 1, structure=structure)
    misclassified = 0
    for label in range(1, num_pred_objects + 1):
        component = (labeled_pred == label)
        if np.any(component & calc_vol) and not np.any(component & targ_vol):
            misclassified += 1
    return pred_nlabels, targ_nlabels, n_overlaps, misclassified

def compute_metrics(pred, targ):
    structure = np.ones((3, 3, 3), dtype=bool)
    total_pred_marker_count = 0
    total_targ_marker_count = 0
    total_overlap_count = 0
    total_misclassified = 0
    # Create binary masks for markers (class==1) and calcifications (class==2)
    pred_marker = (pred == 1).astype(np.int32)
    targ_marker = (targ == 1).astype(np.int32)
    targ_calc = (targ == 2).astype(np.int32)
    pred_marker = scipy.ndimage.binary_dilation(pred_marker)
    targ_marker = scipy.ndimage.binary_dilation(targ_marker)
    targ_calc = scipy.ndimage.binary_dilation(targ_calc)
    for i in range(pred_marker.shape[0]):
        p_vol = pred_marker[i]
        t_vol = targ_marker[i]
        c_vol = targ_calc[i]
        p_n, t_n, n_overlap, misclassified = process_volume(p_vol, t_vol, c_vol, structure)
        total_pred_marker_count += p_n
        total_targ_marker_count += t_n
        total_overlap_count += n_overlap
        total_misclassified += misclassified
    false_negative = total_targ_marker_count - total_overlap_count
    false_positive = total_pred_marker_count - total_overlap_count
    return {
         "actual_markers": total_targ_marker_count,
         "true_positive": total_overlap_count,
         "false_negative": false_negative,
         "false_positive": false_positive,
         "misclassified": total_misclassified
    }

# Hyperparameters for prior losses
lambda_count = 0.001
lambda_size = 0.001
lambda_round = 0.001

# --- Prepare lists for loss plotting ---
train_loss_list = []
valid_loss_list = []

# --- Training loop with metrics printing and loss graph saving ---
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_epochs)

best_valid_loss = float('inf')
epochs_no_improve = 0
patience = 300

# Define column names
log_filename = f"{model_instance}-training.csv"
columns = [
    "Epoch", "Set", "CE", "Dice", "λ₁ ⋅ CountLoss", "λ₂ ⋅ SizeLoss", "λ₃ ⋅ RoundLoss",
    "Total Loss", "Markers Found", "TPs", "FNs", "FPs", "Misclassified Calcs"
]
df_log = pd.DataFrame(columns=columns)

print("Starting training loop...")
start_time = time.time()
for epoch in range(training_epochs):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        base_loss = combined_loss(outputs, masks)
        count_loss, size_loss, round_loss = compute_prior_losses(outputs)
        prior_loss = lambda_count * count_loss + lambda_size * size_loss + lambda_round * round_loss
        total_loss = base_loss + torch.tensor(prior_loss, device=device, dtype=torch.float32)
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)
    
    model.eval()
    valid_loss = 0
    all_preds = []
    all_targs = []
    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            base_loss = combined_loss(outputs, masks)
            count_loss, size_loss, round_loss = compute_prior_losses(outputs)
            prior_loss = lambda_count * count_loss + lambda_size * size_loss + lambda_round * round_loss
            total_loss = base_loss + torch.tensor(prior_loss, device=device, dtype=torch.float32)
            valid_loss += total_loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targs.append(masks.cpu().numpy())
    valid_loss /= len(valid_loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targs = np.concatenate(all_targs, axis=0)
    
    metrics = compute_metrics(all_preds, all_targs)

    # Format results for DataFrame
    new_rows = [
        [epoch+1, "Train", ce_loss_fn(outputs, masks).item(), dice_loss(outputs, masks).item(),
         lambda_count * count_loss, lambda_size * size_loss, lambda_round * round_loss,
         train_loss, metrics['actual_markers'], metrics['true_positive'],
         metrics['false_negative'], metrics['false_positive'], metrics['misclassified']],
        
        [epoch+1, "Validation", ce_loss_fn(outputs, masks).item(), dice_loss(outputs, masks).item(),
         lambda_count * count_loss, lambda_size * size_loss, lambda_round * round_loss,
         valid_loss, metrics['actual_markers'], metrics['true_positive'],
         metrics['false_negative'], metrics['false_positive'], metrics['misclassified']]
    ]

    # Append new data
    df_log = pd.concat([df_log, pd.DataFrame(new_rows, columns=columns)], ignore_index=True)

    # Save updated DataFrame to CSV
    df_log.to_csv(log_filename, index=False)

    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    
    # Plot and save the loss graph (updates each epoch)
    plt.figure()
    plt.plot(range(1, epoch+2), train_loss_list, label="Train Loss")
    plt.plot(range(1, epoch+2), valid_loss_list, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1)
    plt.title(f"Loss Curve for {model_instance}")
    plt.legend()
    plt.savefig(f"{model_instance}-loss.png")
    plt.close()
    
    if valid_loss < best_valid_loss - 0.01:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f"{model_instance}-best.pth")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= patience:
        print("Early stopping triggered")
        break
    scheduler.step()

end_time = time.time()
duration_mins = (end_time - start_time) / 60
print(f"Finished training after {round(duration_mins, 2)} mins")
torch.save(model.state_dict(), f"{model_instance}-final.pth")

