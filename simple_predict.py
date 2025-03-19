import argparse
import os
import torch
import nibabel as nib
import torchio as tio
import glob
import numpy as np
from sklearn.model_selection import KFold

import fastMONAI.vision_all
from fastMONAI.vision_all import (
    MedDataBlock, ImageBlock, MedImage,
    ColReader, PadOrCrop, ZNormalization,
    DataLoaders, IndexSplitter
)

from monai.networks.nets import UNet
from monai.losses import DiceCELoss
import scipy.ndimage as ndimage

def pad_or_crop_tensor(input_tensor, target_size, interpolation='trilinear'):
    # Calculate padding or cropping values for depth, height and width
    diff_depth = target_size[0] - input_tensor.shape[0]
    diff_height = target_size[1] - input_tensor.shape[1]
    diff_width = target_size[2] - input_tensor.shape[2]

    # Handle depth
    if diff_depth > 0:
        pad_depth = diff_depth
        crop_depth = slice(None)
    else:
        pad_depth = 0
        crop_depth = slice(-diff_depth // 2, diff_depth // 2 + input_tensor.shape[0])

    # Handle height
    if diff_height > 0:
        pad_height = diff_height
        crop_height = slice(None)
    else:
        pad_height = 0
        crop_height = slice(-diff_height // 2, diff_height // 2 + input_tensor.shape[1])

    # Handle width
    if diff_width > 0:
        pad_width = diff_width
        crop_width = slice(None)
    else:
        pad_width = 0
        crop_width = slice(-diff_width // 2, diff_width // 2 + input_tensor.shape[2])

    # Apply padding if needed
    if pad_depth > 0 or pad_height > 0 or pad_width > 0:
        input_tensor = torch.nn.functional.pad(
            input_tensor, 
            (pad_width//2, pad_width//2, pad_height//2, pad_height//2, pad_depth//2, pad_depth//2)
        )

    # Crop tensor if needed
    input_tensor = input_tensor[crop_depth, crop_height, crop_width]

    return input_tensor

def resample_to_1mm(input_nifti: str, output_nifti: str) -> str:
    """
    Load a NIfTI file, resample it to 1 mm isotropic, and save as <original>_resampled.nii.
    """
    img = tio.ScalarImage(input_nifti)
    resample_transform = tio.Resample((1, 1, 1))
    resampled = resample_transform(img)
    data_4d = resampled.data
    affine = resampled.affine
    data_3d = data_4d.squeeze(0).numpy()
    out_nii = nib.Nifti1Image(data_3d, affine)
    nib.save(out_nii, output_nifti)
    print(f"Saved resampled image to {output_nifti}")
    return output_nifti

def main():
    parser = argparse.ArgumentParser(description="Minified inference to save raw label maps + final segmentation.")
    parser.add_argument("--model", help="Path to the trained .pth model (optional)")
    parser.add_argument("--model_type", default='T1')
    parser.add_argument("--input", required=True, help="Path to the input NIfTI file")
    parser.add_argument("--output", required=True, help="Directory to save outputs")
    parser.add_argument("--shape", nargs=3, type=int, default=[80, 80, 80],
                        help="PadOrCrop shape, e.g. --shape 224 196 50")
    parser.add_argument("--clean", action="store_true",
                        help="Run clean_up_segmentation before saving final seg")
    args = parser.parse_args()

    # If no model is provided and the input path contains a 'bids' folder,
    # extract the subject name and determine the fold index.
    if not args.model:
        input_parts = os.path.abspath(args.input).split(os.sep)
        if "bids" in input_parts:
            bids_index = input_parts.index("bids")
            try:
                subject_name = input_parts[bids_index + 1]
            except IndexError:
                print("Could not find subject folder after 'bids'.")
                exit(1)
        else:
            print("--model parameter not provided and no 'bids' folder found in input path.")
            exit(1)

        bids_dir = os.path.sep + os.path.join(*input_parts[:bids_index + 1])
        subject_dirs = [d for d in os.listdir(bids_dir) if d.startswith("sub-") and os.path.isdir(os.path.join(bids_dir, d))]
        if subject_name not in subject_dirs:
            print(f"Subject {subject_name} not found in bids directory.")
            exit(1)

        subjects = sorted(subject_dirs)
        n_splits = min(25, len(subjects))
        kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
        subjects_array = np.array(subjects)
        fold_index = None
        for i, (train_idx, val_idx) in enumerate(kf.split(subjects_array)):
            if subject_name in subjects_array[val_idx]:
                fold_index = i
                break
        if fold_index is None:
            print(f"Subject {subject_name} not assigned to any fold.")
            exit(1)
        pattern = os.path.abspath(f"models/{args.model_type}-2024*-*-{fold_index}-best*.pth")
        model_files = glob.glob(pattern)
        if not model_files:
            print(f"No model file found matching pattern: {pattern}")
            exit(1)
        args.model = model_files[0]
        print(f"Auto-selected model: {args.model}")

    print("Resampling input...")
    args.input = resample_to_1mm(args.input, os.path.join(args.output, 'resampled.nii'))

    # 1) Load the model architecture and weights.
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    )
    learn = fastMONAI.vision_all.Learner(
        dls=None,
        model=model,
        loss_func=DiceCELoss(
            to_onehot_y=True,
            include_background=True,
            softmax=True
        )
    )
    learn.load(args.model.replace(".pth", ""))
    if torch.cuda.is_available():
        learn.model.cuda()

    # 2) Build a data block to replicate the inference transforms.
    eval_augmentations = [
        PadOrCrop(args.shape),
        ZNormalization(),
    ]
    df = fastMONAI.vision_all.pd.DataFrame({"img_files": [args.input]})
    dblock = fastMONAI.vision_all.MedDataBlock(
        blocks=(ImageBlock(cls=MedImage),),
        splitter=IndexSplitter([]),
        get_x=ColReader("img_files"),
        item_tfms=eval_augmentations
    )
    dls = dblock.dataloaders(df, bs=1)

    # 3) Run inference on the single item.
    (x,) = next(iter(dls.train))
    if torch.cuda.is_available():
        x = x.cuda()

    with torch.no_grad():
        pred = learn.model(x)  # shape: (1, 3, D, H, W)
    pred_data = pred.cpu()[0]  # shape: (3, D, H, W)

    # 4) Extract each channel and final segmentation.
    pred_empty = pred_data[0]
    pred_seed  = pred_data[1]
    pred_calc  = pred_data[2]
    pred_seg = torch.argmax(pred_data, dim=0)

    # 5) Save each volume using the original affine and header.
    os.makedirs(args.output, exist_ok=True)
    ref_nii = nib.load(args.input)
    def save_vol(tensor_3d, out_name, interpolation='trilinear'):
        tensor_3d = pad_or_crop_tensor(tensor_3d, ref_nii.shape, interpolation=interpolation)
        out_path = os.path.join(args.output, out_name)
        out_img = nib.Nifti1Image(tensor_3d.numpy(), ref_nii.affine, ref_nii.header)
        nib.save(out_img, out_path)
        print(f"Saved: {out_path}")

    save_vol(pred_empty, "pred_empty.nii")
    save_vol(pred_seed,  "pred_seed.nii")
    save_vol(pred_calc,  "pred_calc.nii")
    save_vol(pred_seg,   "pred_seg.nii", interpolation='nearest')

if __name__ == "__main__":
    main()

# GOOD MODELS SO FAR ON 
# bids-2025/sub-z0394025/ses-20240930/anat/sub-z0394025_ses-20240930_acq-t1tragradientseed_T1w.nii
# /home/ashley/mount/T1-20240405-062011-3-final/pred_seg.nii
# /home/ashley/mount/T1-20240405-071429-7-final/pred_seg.nii
# /home/ashley/mount/T1-20240405-080940-13-final/pred_seg.nii