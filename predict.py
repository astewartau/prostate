# %%
import glob
import os
import sys
import json

import torch
import pandas as pd
import numpy as np
import nibabel as nib

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import fastMONAI.vision_all
from monai.networks.nets import UNet
from monai.losses import DiceCELoss

from useful_functions import *
from utils import *

# %%
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

# %% 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU is available. Fastai will use the GPU.")
else:
    device = torch.device("cpu")
    print("GPU is NOT available. Fastai will use the CPU.")

# %%
bids_dir = "bids"

# %%
session_dirs = []
for json_path in sorted(glob.glob(os.path.join(bids_dir, "sub*", "ses*", "anat", "*echo-01*mag*json"))):
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
        if json_data['ProtocolName'] in ["wip_iSWI_fl3d_vibe_TRY THIS ONE"]:#, "wip_iSWI_fl3d_vibe", "wip_iSWI_fl3d_vibe_TRY THIS ONE"]:
            session_dirs.append(os.sep.join(os.path.split(json_path)[0].split(os.sep)[:-1]))
print(f"{len(session_dirs)} sessions found.")

# remove all sessions with 'sub-z0449294' in the path
session_dirs = sorted(set([s for s in session_dirs if 'sub-z0449294' not in s]))

print(f"{len(session_dirs)} sessions found.")

# %%
extra_files = sum((glob.glob(os.path.join(session_dir, "extra_data", "*.nii*")) for session_dir in session_dirs), [])

qsm_files2 = sorted(sum((glob.glob(os.path.join(session_dir, "extra_data", "*real.nii*")) for session_dir in session_dirs), []))
qsm_files = [x for x in sorted(glob.glob(os.path.join("out/qsm/*.nii"))) if 'sub-z0449294' not in x]
t2s_files = [x for x in sorted(glob.glob(os.path.join("out/t2s/*.nii"))) if 'sub-z0449294' not in x]
r2s_files = [x for x in sorted(glob.glob(os.path.join("out/r2s/*.nii"))) if 'sub-z0449294' not in x]
swi_files = [x for x in sorted(glob.glob(os.path.join("out/swi/*swi.nii"))) if 'sub-z0449294' not in x]
mag_files = [x for x in sorted(sum((glob.glob(os.path.join(session_dir, "extra_data", "magnitude_combined.nii")) for session_dir in session_dirs), [])) if 'sub-z0449294' not in x]
fmap_files = sorted([extra_file for extra_file in extra_files if 'B0.nii' in extra_file])

gre_seg_files = sorted([extra_file for extra_file in extra_files if all(pattern in extra_file for pattern in ['segmentation_clean.nii', 'tgv'])])
t1_files = sorted([extra_file for extra_file in extra_files if 'T1w_resliced' in extra_file])

# Missing CT for sub-z0449294 
ct_files = sorted([extra_file for extra_file in extra_files if 'resliced.nii' in extra_file and 'T1w' not in extra_file and 'segmentation' not in extra_file])
ct_seg_files = sorted([extra_file for extra_file in extra_files if 'resliced_segmentation.nii' in extra_file and 'T1w' not in extra_file])
ct_seg_clean_files = sorted([extra_file for extra_file in extra_files if 'resliced_segmentation_clean.nii' in extra_file and 'T1w' not in extra_file])

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

# %%
assert(len(qsm_files) == len(gre_seg_files))
assert(len(qsm_files) == len(t2s_files))
assert(len(qsm_files) == len(r2s_files))
assert(len(qsm_files) == len(swi_files))
assert(len(qsm_files) == len(mag_files))
assert(len(qsm_files) == len(t1_files))
assert(len(ct_files) == len(ct_seg_clean_files))

# %%
model_data = { 
    'CT' : { 'ct_files': ct_files, 'seg_files': ct_seg_clean_files },
    'QSM' : { 'qsm_files': qsm_files, 'seg_files': gre_seg_files },
    'QSM-SWI' : { 'qsm_files': qsm_files, 'swi_files': swi_files, 'seg_files': gre_seg_files },
    'QSM-T1-R2s' : { 'qsm_files': qsm_files, 't1_files': t1_files, 'r2s_files': r2s_files, 'seg_files': gre_seg_files },
    'QSM-T1' : { 'qsm_files': qsm_files, 't1_files': t1_files, 'seg_files': gre_seg_files },
    'T1' : { 't1_files': t1_files, 'seg_files': gre_seg_files },
    'SWI' : { 'swi_files': swi_files, 'seg_files': gre_seg_files },
    'R2s' : { 'r2s_files': r2s_files, 'seg_files': gre_seg_files },
    'GRE' : { 'mag_files': mag_files, 'seg_files': gre_seg_files },
    'QSM-FMAP': { 'qsm_files': qsm_files, 'fmap_files': fmap_files, 'seg_files': gre_seg_files },
    'FMAP': { 'fmap_files': fmap_files, 'seg_files': gre_seg_files }
}

# %%

for model_name in ['FMAP', 'QSM', 'GRE']:
    k_folds = 25
    random_state = 42
    batch_size = 6
    ce_loss_weights = torch.Tensor([1, 1, 1])
    evaluation_augmentations = [
        fastMONAI.vision_all.PadOrCrop([80, 80, 80]),
        fastMONAI.vision_all.ZNormalization(),
    ]
    subjects = [x.split(os.sep)[1] for x in session_dirs]

    # split training/testing
    df = pd.DataFrame(model_data[model_name])

    infile_cols = [key for key in list(model_data[model_name].keys()) if key != 'seg_files']
    n_input_channels = len(infile_cols)
    print(f"infile_cols: {infile_cols}; n_input_channels: {n_input_channels}")

    # determine resampling suggestion
    med_dataset = fastMONAI.vision_all.MedDataset(
        img_list=df.seg_files.tolist(),
        dtype=fastMONAI.vision_all.MedMask
    )
    suggested_voxelsize, requires_resampling = med_dataset.suggestion()
    largest_imagesize = med_dataset.get_largest_img_size(resample=suggested_voxelsize)

    for session_dir in session_dirs:
        subject_name = session_dir.split(os.sep)[1]

        if not df.applymap(lambda cell: subject_name in str(cell)).any().any():
            print(f"ERROR: Subject {subject_name} not found!")

        marker_precisions = []
        marker_recalls = []
        precisions = []
        recalls = []
        fprs = []
        tprs = []
        fm_losses = []
        calc_losses = []

        kf = list(KFold(n_splits=k_folds, random_state=random_state, shuffle=True).split(df))

        train_index = None
        valid_index = None
        for i in range(len(kf)):
            (train_index_i, valid_index_i) = kf[i]
            if df.iloc[valid_index_i].applymap(lambda cell: subject_name in str(cell)).any().any():
                (train_index, valid_index) = kf[i]
                break
        if valid_index is None:
            print(f"ERROR: Subject {subject_name} not found")
            continue
        else:
            print(f"Subject {subject_name} found in fold {i}")

        dblock = fastMONAI.vision_all.MedDataBlock(
            blocks=(fastMONAI.vision_all.ImageBlock(cls=fastMONAI.vision_all.MedImage), fastMONAI.vision_all.MedMaskBlock),
            splitter=fastMONAI.vision_all.IndexSplitter(valid_index),
            get_x=fastMONAI.vision_all.ColReader(infile_cols),
            get_y=fastMONAI.vision_all.ColReader('seg_files'),
            item_tfms=evaluation_augmentations,
            reorder=requires_resampling,
            resample=suggested_voxelsize
        )

        dls = fastMONAI.vision_all.DataLoaders.from_dblock(dblock, df, bs=batch_size)

        learn = fastMONAI.vision_all.Learner(
            dls,
            model=UNet(
                spatial_dims=3,
                in_channels=n_input_channels,
                out_channels=3,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2
            ),
            loss_func=DiceCELoss(
                to_onehot_y=True,
                include_background=True,
                softmax=True,
                ce_weight=ce_loss_weights
            ),
            opt_func=fastMONAI.vision_all.ranger,
            metrics=[fastMONAI.vision_all.multi_dice_score, MarkersIdentified(), SuperfluousMarkers()]#.to_fp16()
        )

        model_file = glob.glob(f"models/{model_name}-2024*-*-{i}-best*")[0]
        print(model_file)
        learn = learn.load(model_file.replace("models/", "").replace(".pth", ""))

        if torch.cuda.is_available():
            learn.model.cuda()

        # Compute metrics on the entire training dataset
        correct_markers = MarkersIdentified()

        dblock_valid_eval = fastMONAI.vision_all.MedDataBlock(
            blocks=(fastMONAI.vision_all.ImageBlock(cls=fastMONAI.vision_all.MedImage), fastMONAI.vision_all.MedMaskBlock),
            splitter=fastMONAI.vision_all.IndexSplitter([]),
            get_x=fastMONAI.vision_all.ColReader(infile_cols),
            get_y=fastMONAI.vision_all.ColReader('seg_files'),
            item_tfms=evaluation_augmentations,
            reorder=requires_resampling,
            resample=suggested_voxelsize
        )
        dls_valid_eval = fastMONAI.vision_all.DataLoaders.from_dblock(dblock_valid_eval, df.iloc[valid_index], bs=1, sampler=fastMONAI.vision_all.SequentialSampler)
        for x, y in dls_valid_eval.train:
            print(f"{i}/{len(dls_valid_eval.train)}", end="; ")
            pred = torch.argmax(learn.model(x), dim=1).unsqueeze(1).to(dtype=torch.float)
            correct_markers.accumulate(pred=pred.cpu(), targ=y.cpu())

        marker_tps = correct_markers.overlap_count
        marker_fps = correct_markers.pred_marker_count - correct_markers.overlap_count
        marker_fns = correct_markers.targ_marker_count - correct_markers.overlap_count

        marker_precision = (marker_tps / (marker_tps + marker_fps)) if (marker_tps + marker_fps) > 0 else 0
        marker_recall = (marker_tps / (marker_tps + marker_fns))  if (marker_tps + marker_fns) > 0 else 0

        loss, *metrics = learn.validate(ds_idx=0, dl=dls_valid_eval.train)

        # get predictions
        dls_valid_eval = fastMONAI.vision_all.DataLoaders.from_dblock(dblock_valid_eval, df.iloc[valid_index], bs=len(dls_valid_eval.train_ds), sampler=fastMONAI.vision_all.SequentialSampler)
        subject_idx = dls_valid_eval.train.items.loc[valid_index_i][dls_valid_eval.train.items.loc[valid_index_i]['seg_files'].str.contains(subject_name)].index[0]
        subject_i = dls_valid_eval.train.items.index.get_loc(subject_idx)

        x, y = list(dls_valid_eval.train)[0]
        pred = learn.model(x).unsqueeze(1).to(dtype=torch.float)
        pred_data = torch.Tensor(pred)
        pred_empty = pred_data[subject_i,0,0,:,:,:]
        pred_seed = pred_data[subject_i,0,1,:,:,:]
        pred_calc = pred_data[subject_i,0,2,:,:,:]
        pred_seg = torch.argmax(pred_data[subject_i,0,:,:,:], dim=0)

        pred_empty = pad_or_crop_tensor(pred_empty, (224, 196, 50))
        pred_seed = pad_or_crop_tensor(pred_seed, (224, 196, 50))
        pred_calc = pad_or_crop_tensor(pred_calc, (224, 196, 50))
        pred_seg = pad_or_crop_tensor(pred_seg.cpu(), (224, 196, 50), interpolation='nearest')

        nii = nib.load(dls_valid_eval.train.items['seg_files'].loc[subject_idx].split(';')[0])
        extra_data_dir = os.path.join(session_dir, "extra_data")
        nib.save(nib.Nifti1Image(dataobj=pred_seed.cpu().detach().numpy(), affine=nii.affine, header=nii.header), os.path.join(extra_data_dir, f"{subject_name}_{model_name}_pred_seed2.nii"))
        nib.save(nib.Nifti1Image(dataobj=pred_calc.cpu().detach().numpy(), affine=nii.affine, header=nii.header), os.path.join(extra_data_dir, f"{subject_name}_{model_name}_pred_calc2.nii"))
        nib.save(nib.Nifti1Image(dataobj=pred_empty.cpu().detach().numpy(), affine=nii.affine, header=nii.header), os.path.join(extra_data_dir, f"{subject_name}_{model_name}_pred_empty2.nii"))
        nib.save(nib.Nifti1Image(dataobj=pred_seg.cpu().detach().numpy(), affine=nii.affine, header=nii.header), os.path.join(extra_data_dir, f"{subject_name}_{model_name}_pred_seg2.nii"))


# %%
print("DONE")

