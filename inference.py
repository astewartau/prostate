#!/usr/bin/env python
import argparse
import glob
import os
import json
import time
import datetime
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import scipy.ndimage
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import KFold
from skimage.measure import marching_cubes, mesh_surface_area

# ---------------------------
# Define a simple 3D UNet.
# This must match your training architecture.
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

# ---------------------------
# restore_shape() reverses the CropOrPad operation.
def restore_shape(seg, original_shape):
    current_shape = seg.shape
    seg = seg.clone()
    for i in range(3):
        diff = original_shape[i] - current_shape[i]
        if diff > 0:
            pad_before = diff // 2
            pad_after = diff - pad_before
            if i == 0:
                seg = F.pad(seg.unsqueeze(0), (0,0,0,0,pad_before, pad_after)).squeeze(0)
            elif i == 1:
                seg = F.pad(seg.unsqueeze(0), (0,0,pad_before, pad_after,0,0)).squeeze(0)
            elif i == 2:
                seg = F.pad(seg.unsqueeze(0), (pad_before, pad_after,0,0,0,0)).squeeze(0)
        elif diff < 0:
            crop_before = (-diff) // 2
            crop_after = crop_before + original_shape[i]
            if i == 0:
                seg = seg[crop_before:crop_after, :, :]
            elif i == 1:
                seg = seg[:, crop_before:crop_after, :]
            elif i == 2:
                seg = seg[:, :, crop_before:crop_after]
    return seg

# ---------------------------
# Utility: parse model type and fold from model filename.
def parse_model_info(model_path):
    basename = os.path.basename(model_path)
    parts = basename.split('-')
    try:
        model_type = parts[0]
        fold = int(parts[3])
        return model_type, fold
    except Exception as e:
        raise ValueError("Could not parse model type and fold number from filename.") from e

# ---------------------------
# Utility: build model_data dictionary for validation selection.
# For demonstration, this is implemented for T1. Extend as needed.
def get_model_data(model_type):
    bids_dir = "bids"
    session_dirs = []
    for json_path in sorted(glob.glob(os.path.join(bids_dir, "sub*", "ses*", "anat", "*echo-01*mag*json"))):
        with open(json_path, 'r') as jf:
            json_data = json.load(jf)
            if json_data['ProtocolName'] in ["wip_iSWI_fl3d_vibe_TRY THIS ONE"]:
                session_dirs.append(os.sep.join(os.path.split(json_path)[0].split(os.sep)[:-1]))
    session_dirs = sorted(set([s for s in session_dirs if 'sub-z0449294' not in s]))
    extra_files = sum((glob.glob(os.path.join(session_dir, "extra_data", "*.nii*"))
                       for session_dir in session_dirs), [])
    if model_type == "T1":
        files = sorted([ef for ef in extra_files if 'T1w_resliced' in ef and 'homogeneity-corrected' in ef])
        seg_files = sorted([ef for ef in extra_files if all(p in ef for p in ['segmentation_clean.', 'tgv'])])
        return pd.DataFrame({"t1_files": files, "seg_files": seg_files})
    else:
        raise ValueError("Validation input selection for model type {} not implemented.".format(model_type))

# ---------------------------
# Utility: get validation input file for a given fold.
def get_validation_input_from_fold(model_type, fold, random_state=42):
    df = get_model_data(model_type)
    qsm_files = [x for x in sorted(glob.glob(os.path.join("out/qsm/*.nii")))
                 if 'sub-z0449294' not in x]
    n_splits = len(qsm_files)
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    splits = list(kf.split(df))
    if fold >= len(splits):
        raise ValueError(f"Fold {fold} out of range for n_splits={n_splits}.")
    _, valid_index = splits[fold]
    val_idx = valid_index[0]
    input_file = df.iloc[val_idx]['t1_files']
    return input_file

# ---------------------------
# Loss and metric helper functions (adapted from training)
ce_loss_fn = nn.CrossEntropyLoss()
def dice_loss(pred, target, eps=1e-5):
    pred_soft = F.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
    intersection = (pred_soft * target_onehot).sum(dim=(2,3,4))
    sums = pred_soft.sum(dim=(2,3,4)) + target_onehot.sum(dim=(2,3,4))
    dice = (2 * intersection + eps) / (sums + eps)
    return 1 - dice.mean()
def combined_loss(pred, target):
    return ce_loss_fn(pred, target) + dice_loss(pred, target)

def compute_prior_losses(outputs):
    # Compute prior losses using predicted segmentation (assumes class 1 is marker)
    segmentation = torch.argmax(outputs, dim=1).detach().cpu().numpy()
    B = segmentation.shape[0]
    batch_count_loss = 0
    batch_size_loss = 0
    batch_round_loss = 0
    for i in range(B):
        binary_mask = (segmentation[i] == 1).astype(np.uint8)
        labeled, num_components = scipy.ndimage.label(binary_mask)
        count_loss = (num_components - 3)**2
        size_loss = 0
        round_loss = 0
        for comp in range(1, num_components + 1):
            comp_mask = (labeled == comp).astype(np.uint8)
            vol = comp_mask.sum()
            if vol < 15:
                size_penalty = (15 - vol)**2
            elif vol > 80:
                size_penalty = (vol - 80)**2
            else:
                size_penalty = 0
            size_loss += size_penalty
            try:
                verts, faces, normals, values = marching_cubes(comp_mask.astype(np.float32), level=0.5)
                area = mesh_surface_area(verts, faces)
                sphericity = (np.pi**(1/3) * (6 * vol)**(2/3)) / area
                round_penalty = (1 - sphericity)**2
            except Exception as e:
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
    # Binary masks: assume class==1 are markers and class==2 are calcifications
    pred_marker = (pred == 1).astype(np.int32)
    targ_marker = (targ == 1).astype(np.int32)
    targ_calc = (targ == 2).astype(np.int32)
    pred_marker = scipy.ndimage.binary_dilation(pred_marker)
    targ_marker = scipy.ndimage.binary_dilation(targ_marker)
    targ_calc = scipy.ndimage.binary_dilation(targ_calc)
    # Process slice by slice
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

# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Run segmentation on a NIfTI file using a pretrained model.")
    parser.add_argument('--input', type=str, help="Path to the input NIfTI file (if omitted, automatically selects validation input)")
    parser.add_argument('--model', type=str, required=True, help="Path to the model .pth file")
    parser.add_argument('--output', type=str, required=True, help="Path to the output NIfTI segmentation file")
    args = parser.parse_args()

    model_type, fold_from_filename = parse_model_info(args.model)

    if not args.input:
        args.input = get_validation_input_from_fold(model_type, fold_from_filename)

    nii = nib.load(args.input)
    input_data = nii.get_fdata()
    original_shape = input_data.shape  # (D, H, W)
    affine = nii.affine

    # Save the original input for reference.
    input_outfile = args.output.replace('.nii', '_input.nii')
    nib.save(nib.Nifti1Image(input_data.astype(np.float32), affine), input_outfile)

    # Create a TorchIO ScalarImage (shape: (1, D, H, W))
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=torch.tensor(input_data, dtype=torch.float32).unsqueeze(0), affine=affine)
    )
    # Pre-process: ZNormalization and CropOrPad to (80,80,80)
    transform = tio.Compose([
        tio.ZNormalization(),
        tio.CropOrPad((80,80,80))
    ])
    subject_transformed = transform(subject)
    input_tensor = subject_transformed['image'].data  # shape (1,80,80,80)

    # Prepare model (assumed architecture must match training)
    model = UNet3D(in_channels=1, out_channels=3)
    model.to('cpu')
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        input_tensor = input_tensor.unsqueeze(0)  # shape: (1,1,80,80,80)
        outputs = model(input_tensor)             # shape: (1,3,80,80,80)
        preds = torch.argmax(outputs, dim=1)        # shape: (1,80,80,80)
    preds = preds.squeeze(0)  # shape: (80,80,80)

    # Restore segmentation to original shape (using symmetric padding/cropping)
    seg_restored = restore_shape(preds, original_shape)

    # Save the segmentation (as int16) with original affine.
    seg_outfile = args.output
    out_nii = nib.Nifti1Image(seg_restored.numpy().astype(np.int16), affine)
    nib.save(out_nii, seg_outfile)

    # Optionally, also save the raw outputs (probabilities/logits) with a suffix.
    outputs_outfile = args.output.replace('.nii', '_outputs.nii')
    out_background = restore_shape(outputs[0,0].squeeze(0), original_shape)
    out_markers = restore_shape(outputs[0,1].squeeze(0), original_shape)
    out_calcification = restore_shape(outputs[0,2].squeeze(0), original_shape)
    out_logits = torch.stack([out_background, out_markers, out_calcification], dim=3)
    out_nii_logits = nib.Nifti1Image(out_logits.numpy().astype(np.float32), affine)
    nib.save(out_nii_logits, outputs_outfile)

    # ---------------------------
    # If a matching ground-truth segmentation exists (i.e. input was auto-selected),
    # load it, apply the same cropping transform (without normalization) and compute losses & metrics.
    df_model = get_model_data(model_type)
    row = df_model[df_model['t1_files'] == args.input]
    if not row.empty:
        gt_seg_file = row.iloc[0]['seg_files']
        gt_nii = nib.load(gt_seg_file)
        gt_seg_data = gt_nii.get_fdata()
        # Create a TorchIO LabelMap for segmentation; only CropOrPad is applied.
        subject_seg = tio.Subject(
            seg=tio.LabelMap(tensor=torch.tensor(gt_seg_data, dtype=torch.int16).unsqueeze(0), affine=gt_nii.affine)
        )
        transform_seg = tio.CropOrPad((80,80,80))
        subject_seg_transformed = transform_seg(subject_seg)
        gt_seg_tensor = subject_seg_transformed['seg'].data.squeeze(0).long()  # shape (80,80,80)

        # Compute loss values using the transformed outputs and ground truth.
        # (Add a batch dimension to gt_seg_tensor to match outputs shape.)
        ce_loss_value = ce_loss_fn(outputs, gt_seg_tensor.unsqueeze(0)).item()
        dice_loss_value = dice_loss(outputs, gt_seg_tensor.unsqueeze(0)).item()
        base_loss_value = ce_loss_value + dice_loss_value
        count_loss, size_loss, round_loss = compute_prior_losses(outputs)
        prior_loss_value = lambda_count * count_loss + lambda_size * size_loss + lambda_round * round_loss
        total_loss_value = base_loss_value + prior_loss_value

        # Compute metrics using the predicted segmentation (on the transformed space).
        preds_tensor = torch.argmax(outputs, dim=1)  # shape: (1,80,80,80)
        preds_np = preds_tensor.cpu().numpy()[0]
        gt_np = gt_seg_tensor.cpu().numpy()
        metrics = compute_metrics(preds_np[np.newaxis, ...], gt_np[np.newaxis, ...])

        # Get subject name from args.input (e.g. filepath is bids/sub-z3268423/ses-20240109/extra_data/T1.nii sub is sub-z3268423)
        subject_name = [dir_name for dir_name in args.input.split(os.sep) if dir_name.startswith('sub-')][0]

        # Create a dataframe with the results (displaying floats to 4 decimals).
        data = {
            "Set": ["Validation"],
            "Subject": [subject_name],
            "CE": [f"{ce_loss_value:.4f}"],
            "Dice": [f"{dice_loss_value:.4f}"],
            "λ₁ ⋅ CountLoss": [f"{lambda_count * count_loss:.4f}"],
            "λ₂ ⋅ SizeLoss": [f"{lambda_size * size_loss:.4f}"],
            "λ₃ ⋅ RoundLoss": [f"{lambda_round * round_loss:.4f}"],
            "Total Loss": [f"{total_loss_value:.4f}"],
            "Markers Found": [metrics['actual_markers']],
            "TPs": [metrics['true_positive']],
            "FNs": [metrics['false_negative']],
            "FPs": [metrics['false_positive']],
            "Misclassified Calcs": [metrics['misclassified']]
        }
        df_results = pd.DataFrame(data)
        print(df_results.T.rename(columns={0: "Value"}), flush=True)

        
    else:
        print("No matching ground-truth segmentation found for the input file.", flush=True)

if __name__ == '__main__':
    main()
