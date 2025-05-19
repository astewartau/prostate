#!/usr/bin/env python
import argparse
import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import nibabel as nib
from sklearn.model_selection import KFold
import scipy.ndimage
import matplotlib.pyplot as plt
import datetime
import time
import sys

# UNet3D model definition
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

# Custom transform: Merge Input Channels
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
        if 'seg' in subject:
            subject['mask'] = subject['seg']
        return subject

# Build dataframe from bids directories
def build_dataframe(bids_dirs):
    paths_list = []
    for bids_dir in bids_dirs:
        subject_dirs = sorted(glob.glob(os.path.join(bids_dir, "sub-*")))
        subject_dirs = [s for s in subject_dirs if 'sub-z0449294' not in s]
        for subject_dir in subject_dirs:
            paths_dict = {}
            nii_paths = glob.glob(os.path.join(subject_dir, "ses-*", "*", "*.nii"))
            for nii_path in nii_paths:
                modality_name = os.path.basename(nii_path).split('.')[0]
                if modality_name.startswith('sub-'):
                    continue
                paths_dict[modality_name] = nii_path
            paths_list.append(paths_dict)
    df = pd.DataFrame(paths_list)
    df = df.where(pd.notnull(df), None)
    return df

# Parse model filename to extract model name and fold ID.
def parse_model_filename(model_filepath):
    base = os.path.basename(model_filepath)
    parts = base.split('-')
    if len(parts) < 5:
        raise ValueError("Unexpected model filename format.")
    model_name = parts[0]
    fold_id = int(parts[-2])
    return model_name, fold_id

# Helper: Match shapes of two arrays by cropping to the smallest dimensions.
def match_shapes(a, b):
    new_shape = tuple(min(a_dim, b_dim) for a_dim, b_dim in zip(a.shape, b.shape))
    a_cropped = a[tuple(slice(0, s) for s in new_shape)]
    b_cropped = b[tuple(slice(0, s) for s in new_shape)]
    return a_cropped, b_cropped

# Metric functions (using connected components in 3D)
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
    if pred.shape != targ.shape:
        print("Warning: predicted segmentation and ground truth shapes do not match. Cropping to minimum shape.")
        pred, targ = match_shapes(pred, targ)
    structure = np.ones((3, 3, 3), dtype=bool)
    pred_marker = (pred == 1).astype(np.int32)
    targ_marker = (targ == 1).astype(np.int32)
    targ_calc = (targ == 2).astype(np.int32)

    # Compute actual calcification regions from the original target calcifications.
    _, actual_calcifications = scipy.ndimage.label(targ_calc, structure=structure)

    # Apply binary dilation for marker matching.
    pred_marker = scipy.ndimage.binary_dilation(pred_marker)
    targ_marker = scipy.ndimage.binary_dilation(targ_marker)
    targ_calc = scipy.ndimage.binary_dilation(targ_calc)
    p_n, t_n, n_overlap, misclassified = process_volume(pred_marker, targ_marker, targ_calc, structure)
    false_negative = t_n - n_overlap
    false_positive = p_n - n_overlap
    return {
         "actual_markers": t_n,
         "true_positive": n_overlap,
         "false_negative": false_negative,
         "false_positive": false_positive,
         "misclassified": misclassified,
         "actual_calcifications": actual_calcifications
    }

# Load a model from file
def load_model(model_filepath, in_channels, num_classes, device):
    model_name, _ = parse_model_filename(model_filepath)
    print(f"Loading model '{model_name}' from {model_filepath}")
    model = UNet3D(in_channels=in_channels, out_channels=num_classes)
    model = model.to(device)
    state_dict = torch.load(model_filepath, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    return model

# Prepare a subject given a dataframe row
def prepare_subject(row, infile_cols, segmentation_col, transforms):
    subject_dict = {}
    for col in infile_cols:
        if col in row:
            subject_dict[col] = tio.ScalarImage(row[col])
    subject_dict['seg'] = tio.LabelMap(row[segmentation_col])
    subject = tio.Subject(**subject_dict)
    subject = transforms(subject)
    return subject

# Run inference on a single subject
def run_inference_on_subject(model, subject, device):
    with torch.no_grad():
        input_tensor = subject['image'].data.unsqueeze(0).to(device)
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        segmentation = torch.argmax(probs, dim=1)
    seg_np = segmentation.squeeze(0).cpu().numpy().astype(np.uint8)
    gt_tensor = subject['mask'].data.squeeze(0).cpu().numpy().astype(np.uint8)
    metrics = compute_metrics(seg_np, gt_tensor)
    return seg_np, probs, metrics

# Evaluate on the entire dataset
def evaluate_full_dataset(df, model, infile_cols, segmentation_col, transforms, device):
    summary_list = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        subject = prepare_subject(row, infile_cols, segmentation_col, transforms)
        seg_np, probs, metrics = run_inference_on_subject(model, subject, device)
        summary_list.append({
            "Subject Index": idx,
            "Input image path": row[infile_cols[0]],
            "Actual markers": metrics["actual_markers"],
            "True positives": metrics["true_positive"],
            "False negatives": metrics["false_negative"],
            "False positives": metrics["false_positive"],
            "Misclassified": metrics["misclassified"],
            "Actual calcifications": metrics["actual_calcifications"]
        })
        print(f"Processed subject index: {idx}")
    summary_df = pd.DataFrame(summary_list)
    print("\nSummary Metrics for evaluation on the entire dataset:")
    print(summary_df.to_string(index=False))

    total_actual = summary_df["Actual markers"].sum()
    total_true_positive = summary_df["True positives"].sum()
    total_false_negative = summary_df["False negatives"].sum()
    total_false_positive = summary_df["False positives"].sum()
    total_misclassified = summary_df["Misclassified"].sum()
    total_calcifications = summary_df["Actual calcifications"].sum()

    print("\nAggregated Metrics Across All Subjects:")
    print(f"Total actual markers: {total_actual}")
    print(f"Total true positive markers: {total_true_positive}")
    print(f"Total false negative markers: {total_false_negative}")
    print(f"Total false positive markers: {total_false_positive}")
    print(f"Total misclassified markers: {total_misclassified}")
    print(f"Total actual calcifications: {total_calcifications}")

    if total_actual > 0:
        tp_pct = total_true_positive / total_actual * 100
        fn_pct = total_false_negative / total_actual * 100
        fp_pct = total_false_positive / total_actual * 100
        mis_pct = total_misclassified / total_actual * 100
        print("\nPercentage Metrics (relative to actual markers):")
        print(f"True positive percentage: {tp_pct:.2f}%")
        print(f"False negative percentage: {fn_pct:.2f}%")
        print(f"False positive percentage: {fp_pct:.2f}%")
        print(f"Misclassified percentage: {mis_pct:.2f}%")

# Evaluate LOOCV; this function runs the leave-one-out evaluation across all provided models,
# prints per-fold outputs, and builds a summary dataframe with aggregated metrics.
def evaluate_loocv(df, model_filepaths, infile_cols, segmentation_col, num_classes, transforms, device, save_outputs):
    fold_summaries = []
    kf = KFold(n_splits=len(df), random_state=42, shuffle=True)
    splits = list(kf.split(df))
    for model_filepath in model_filepaths:
        model_name, fold_id = parse_model_filename(model_filepath)
        print(f"\nProcessing model: {model_filepath}")
        print(f"Model: {model_name}, Fold ID: {fold_id}")
        if fold_id >= len(splits):
            raise ValueError(f"Fold ID {fold_id} out of range. There are only {len(splits)} folds.")
        _, valid_index = splits[fold_id]
        if len(valid_index) != 1:
            raise ValueError(f"Expected one validation subject, got {len(valid_index)}.")
        val_subject_idx = valid_index[0]
        row = df.iloc[val_subject_idx]
        print(f"Running inference on subject index: {val_subject_idx}")
        subject = prepare_subject(row, infile_cols, segmentation_col, transforms)
        model = load_model(model_filepath, in_channels=len(infile_cols), num_classes=num_classes, device=device)
        seg_np, probs, metrics = run_inference_on_subject(model, subject, device)
        for col in infile_cols:
            if col in row:
                print(f"Input image path: {row[col]}")
        print("Inference Metrics:")
        print(f"  Actual markers: {metrics['actual_markers']}")
        print(f"  True positive markers: {metrics['true_positive']}")
        print(f"  False negative markers: {metrics['false_negative']}")
        print(f"  False positive markers: {metrics['false_positive']}")
        print(f"  Misclassified markers: {metrics['misclassified']}")
        print(f"  Actual calcifications: {metrics['actual_calcifications']}")
        if save_outputs:
            save_nifti_outputs(model_filepath, subject, seg_np, probs)
        summary_dict = {
            "Model path": model_filepath,
            "Fold ID": fold_id,
            "Input image path": row[infile_cols[0]],
            "Actual markers": metrics["actual_markers"],
            "True positives": metrics["true_positive"],
            "False negatives": metrics["false_negative"],
            "False positives": metrics["false_positive"],
            "Misclassified": metrics["misclassified"],
            "Actual calcifications": metrics["actual_calcifications"]
        }
        fold_summaries.append(summary_dict)
    if fold_summaries:
        summary_df = pd.DataFrame(fold_summaries)
        print("\nSummary Metrics for LOOCV evaluation:")
        print(summary_df.to_string(index=False))

        total_actual = summary_df["Actual markers"].sum()
        total_true_positive = summary_df["True positives"].sum()
        total_false_negative = summary_df["False negatives"].sum()
        total_false_positive = summary_df["False positives"].sum()
        total_misclassified = summary_df["Misclassified"].sum()
        total_calcifications = summary_df["Actual calcifications"].sum()

        print("\nAggregated Metrics Across All Models:")
        print(f"Total actual markers: {total_actual}")
        print(f"Total true positive markers: {total_true_positive}")
        print(f"Total false negative markers: {total_false_negative}")
        print(f"Total false positive markers: {total_false_positive}")
        print(f"Total misclassified markers: {total_misclassified}")
        print(f"Total actual calcifications: {total_calcifications}")

        if total_actual > 0:
            tp_pct = total_true_positive / total_actual * 100
            fn_pct = total_false_negative / total_actual * 100
            fp_pct = total_false_positive / total_actual * 100
            mis_pct = total_misclassified / total_actual * 100
            print("\nPercentage Metrics (relative to actual markers):")
            print(f"True positive percentage: {tp_pct:.2f}%")
            print(f"False negative percentage: {fn_pct:.2f}%")
            print(f"False positive percentage: {fp_pct:.2f}%")
            print(f"Misclassified percentage: {mis_pct:.2f}%")

# Save NIfTI outputs
def save_nifti_outputs(model_filepath, subject, seg_np, probs, num_classes):
    base_name = os.path.splitext(os.path.basename(model_filepath))[0]
    seg_img = nib.Nifti1Image(seg_np, subject['image'].affine)
    seg_out_path = f"{base_name}_inference_segmentation.nii.gz"
    nib.save(seg_img, seg_out_path)
    print(f"Saved segmentation to {seg_out_path}")
    probs_np = probs.squeeze(0).cpu().numpy()
    for class_idx in range(num_classes):
        prob_map = probs_np[class_idx, ...]
        prob_img = nib.Nifti1Image(prob_map, subject['image'].affine)
        prob_out_path = f"{base_name}_inference_probability_class_{class_idx}.nii.gz"
        nib.save(prob_img, prob_out_path)
        print(f"Saved probability map for class {class_idx} to {prob_out_path}")
    input_np = subject['image'].data.squeeze(0).cpu().numpy()
    input_img = nib.Nifti1Image(input_np, subject['image'].affine)
    input_out_path = f"{base_name}_inference_input.nii.gz"
    nib.save(input_img, input_out_path)
    print(f"Saved input image to {input_out_path}")
    gt_tensor = subject['mask'].data.squeeze(0).cpu().numpy().astype(np.uint8)
    gt_img = nib.Nifti1Image(gt_tensor, subject['image'].affine)
    gt_out_path = f"{base_name}_ground_truth_segmentation.nii.gz"
    nib.save(gt_img, gt_out_path)
    print(f"Saved ground truth segmentation to {gt_out_path}")

def main():
    parser = argparse.ArgumentParser(description="Perform inference or evaluation using trained models.")
    parser.add_argument("--model", type=str, nargs="+", required=True,
                        help="Path(s) to model .pth file(s)")
    parser.add_argument("--mode", type=str, choices=["full", "loocv"], default="loocv",
                        help="Operation mode: 'full' evaluates the entire dataset, 'loocv' runs leave-one-out evaluation")
    parser.add_argument("--save_outputs", action="store_true",
                        help="Save generated NIfTI outputs")
    args = parser.parse_args()

    bids_dirs = ['bids-2024', 'bids-2025']
    infile_cols = ['t1_corrected']
    segmentation_col = 't1_corrected_segmentation_clean'
    crop_dimensions = (100, 100, 64)
    df = build_dataframe(bids_dirs)
    for col in infile_cols + [segmentation_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe.")
        df = df[df[col].notna()].reset_index(drop=True)
    print(f"Number of subjects in dataframe: {len(df)}")
    if len(df) == 0:
        raise ValueError("No subjects found in the dataframe after filtering.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    in_channels = len(infile_cols)
    num_classes = 3

    common_transforms = tio.Compose([
         tio.CropOrPad(crop_dimensions),
         tio.ZNormalization(),
         MergeInputChannels(infile_cols)
    ])

    if args.mode == "full":
        if len(args.model) != 1:
            raise ValueError("Full dataset evaluation requires a single model.")
        model = load_model(args.model[0], in_channels, num_classes, device)
        evaluate_full_dataset(df, model, infile_cols, segmentation_col, common_transforms, device)
    else:
        evaluate_loocv(df, args.model, infile_cols, segmentation_col, num_classes,
                       common_transforms, device, args.save_outputs)

if __name__ == "__main__":
    main()
