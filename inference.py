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
    print(f"Identified validation input file for fold {fold}: {input_file}", flush=True)
    return input_file

# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Run segmentation on a NIfTI file using a pretrained model.")
    parser.add_argument('--input', type=str, help="Path to the input NIfTI file (if omitted, automatically selects validation input)")
    parser.add_argument('--model', type=str, required=True, help="Path to the model .pth file")
    parser.add_argument('--output', type=str, required=True, help="Path to the output NIfTI segmentation file")
    args = parser.parse_args()

    model_type, fold_from_filename = parse_model_info(args.model)
    print(f"Determined model type: {model_type}, fold: {fold_from_filename}", flush=True)

    if not args.input:
        args.input = get_validation_input_from_fold(model_type, fold_from_filename)
        print(f"Using validation input file: {args.input}", flush=True)
    else:
        print(f"Using provided input file: {args.input}", flush=True)

    print("Loading input NIfTI file:", args.input, flush=True)
    nii = nib.load(args.input)
    input_data = nii.get_fdata()
    original_shape = input_data.shape  # (D, H, W)
    affine = nii.affine

    # Save the original input for reference.
    input_outfile = args.output.replace('.nii', '_input.nii')
    nib.save(nib.Nifti1Image(input_data.astype(np.float32), affine), input_outfile)
    print("Saved original input to:", input_outfile, flush=True)

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
    print("Preprocessed input shape:", input_tensor.shape, flush=True)

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
    print("Model output shape:", preds.shape, flush=True)
    print("Model output unique values:", torch.unique(preds), flush=True)

    # Restore segmentation to original shape (using symmetric padding/cropping)
    seg_restored = restore_shape(preds, original_shape)
    print("Restored segmentation shape:", seg_restored.shape, flush=True)

    # Save the segmentation (as int16) with original affine.
    seg_outfile = args.output
    out_nii = nib.Nifti1Image(seg_restored.numpy().astype(np.int16), affine)
    nib.save(out_nii, seg_outfile)
    print("Saved segmentation to:", seg_outfile, flush=True)

    # Optionally, also save the raw outputs (probabilities/logits) with a suffix.
    outputs_outfile = args.output.replace('.nii', '_outputs.nii')
    # Convert outputs to float32 and squeeze batch dimension.
    
    out_background = restore_shape(outputs[0,0].squeeze(0), original_shape)
    out_markers = restore_shape(outputs[0,1].squeeze(0), original_shape)
    out_calcification = restore_shape(outputs[0,2].squeeze(0), original_shape)
    out_logits = torch.stack([out_background, out_markers, out_calcification], dim=3)
    
    out_nii_logits = nib.Nifti1Image(out_logits.numpy().astype(np.float32), affine)
    nib.save(out_nii_logits, outputs_outfile)
    print("Saved model outputs to:", outputs_outfile, flush=True)

if __name__ == '__main__':
    main()
