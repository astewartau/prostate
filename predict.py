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
from prostate import *

# %%
bids_dir = "bids-new"

# %%
session_dirs = []
for json_path in sorted(glob.glob(os.path.join(bids_dir, "sub*", "ses*", "anat", "*echo-01*mag*json"))):
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
        if json_data['ProtocolName'] == "t2starME_qsm_tra_Iso1.4mm_INPHASE_bipolar_RUN_THIS_ONE":
            session_dirs.append(os.sep.join(os.path.split(json_path)[0].split(os.sep)[:-1]))
print(f"{len(session_dirs)} sessions found")

# %%
extra_files = sum((glob.glob(os.path.join(session_dir, "extra_data", "*.nii*")) for session_dir in session_dirs), [])

qsm_files = sorted(sum((glob.glob(os.path.join(session_dir, "extra_data", "*qsm_echo2-and-echo4.*")) for session_dir in session_dirs), []))
t2s_files = sorted(sum((glob.glob(os.path.join(session_dir, "extra_data", "*t2starmap.nii*")) for session_dir in session_dirs), []))
r2s_files = sorted(sum((glob.glob(os.path.join(session_dir, "extra_data", "*r2starmap.nii*")) for session_dir in session_dirs), []))
mag_files = sorted(sum((glob.glob(os.path.join(session_dir, "extra_data", "magnitude_combined.nii")) for session_dir in session_dirs), []))
swi_files = sorted(sum((glob.glob(os.path.join(session_dir, "extra_data", "swi.nii")) for session_dir in session_dirs), []))

gre_seg_raw_files = [extra_file for extra_file in extra_files if all(pattern in extra_file for pattern in ['_segmentation.', 'run'])]
gre_seg_clean_files = [seg_file.replace(".nii", "_clean.nii") for seg_file in gre_seg_raw_files if os.path.exists(seg_file.replace(".nii", "_clean.nii"))]

t1_files = [extra_file for extra_file in extra_files if any(pattern in extra_file for pattern in ['_T1w', '_t1_tra']) and not any(pattern in extra_file for pattern in ['_Pelvis_', '.json', '_resampled'])]
t1_files = [t1_file.replace("_resampled", "") for t1_file in t1_files]
t1_resampled_files = [t1_file.replace(".nii", "_resampled.nii") for t1_file in t1_files if os.path.exists(t1_file.replace(".nii", "_resampled.nii"))]

ct_files = [extra_file for extra_file in extra_files if 'resliced' in extra_file and any(pattern in extra_file for pattern in ['_na_', '_Pelvis_', '_NA']) and not any(pattern in extra_file for pattern in ['_t1_tra_', 'ATX', 'AXT', 'ROI', 'segmentation', '.json'])]
ct_seg_raw_files = sum((glob.glob(ct_file.replace(".nii", "_segmentation.nii")) for ct_file in ct_files), [])
ct_seg_clean_files = [ct_file.replace("_segmentation", "_segmentation_clean") for ct_file in ct_seg_raw_files if os.path.exists(ct_file)]

print(f"{len(ct_files)} CT images found.")
print(f"{len(ct_seg_raw_files)} raw CT segmentations found.")
print(f"{len(ct_seg_clean_files)} clean CT segmentations found.")
print(f"{len(qsm_files)} QSM images found.")
print(f"{len(mag_files)} magnitude images found.")
print(f"{len(t2s_files)} T2* maps found.")
print(f"{len(r2s_files)} R2* maps found.")
print(f"{len(swi_files)} SWI maps found.")
print(f"{len(t1_files)} T1w files found.")
print(f"{len(t1_resampled_files)} resampled T1w files found.")
print(f"{len(gre_seg_raw_files)} raw GRE segmentations found.")
print(f"{len(gre_seg_clean_files)} clean GRE segmentations found.")

# %%
assert(len(qsm_files) == len(gre_seg_clean_files))
assert(len(qsm_files) == len(t2s_files))
assert(len(qsm_files) == len(r2s_files))
assert(len(qsm_files) == len(swi_files))
assert(len(qsm_files) == len(mag_files))
assert(len(qsm_files) == len(t1_resampled_files))
assert(len(ct_files) == len(ct_seg_clean_files))

# %%
model_data = { # CT, QSM, QSM-T1-R2s, QSM-T1, GRE, T1, SWI, R2s, T2s
    'CT' : { 'in_files' : [f"{ct_files[i]}" for i in range(len(ct_files))], 'seg_files': ct_seg_clean_files },
    'QSM' : { 'in_files' : [f"{qsm_files[i]}" for i in range(len(qsm_files))], 'seg_files': gre_seg_clean_files },
    'QSM-T1-R2s' : { 'in_files' : [f"{qsm_files[i]};{t1_resampled_files[i]};{r2s_files[i]}" for i in range(len(qsm_files))], 'seg_files': gre_seg_clean_files },
    'QSM-T1' : { 'in_files' : [f"{qsm_files[i]};{t1_resampled_files[i]}" for i in range(len(qsm_files))], 'seg_files': gre_seg_clean_files },
    'GRE' : { 'in_files' : [f"{mag_files[i]}" for i in range(len(qsm_files))], 'seg_files': gre_seg_clean_files },
    'T1' : { 'in_files' : [f"{t1_resampled_files[i]}" for i in range(len(qsm_files))], 'seg_files': gre_seg_clean_files },
    'SWI' : { 'in_files' : [f"{swi_files[i]}" for i in range(len(qsm_files))], 'seg_files': gre_seg_clean_files },
    'R2s' : { 'in_files' : [f"{r2s_files[i]}" for i in range(len(qsm_files))], 'seg_files': gre_seg_clean_files },
    'T2s' : { 'in_files' : [f"{t2s_files[i]}" for i in range(len(qsm_files))], 'seg_files': gre_seg_clean_files },
}

# %%
model_name = 'QSM'
subject_name = 'sub-z023'
k_folds = 24
random_state = 42
batch_size = 6
ce_loss_weights = torch.Tensor([1, 1, 1])
evaluation_augmentations = [
    fastMONAI.vision_all.PadOrCrop([80, 80, 80]),
    fastMONAI.vision_all.ZNormalization(),
]

# split training/testing
df = pd.DataFrame(model_data[model_name])

# %%
if not df.applymap(lambda cell: subject_name in str(cell)).any().any():
    print(f"ERROR: Subject {subject_name} not found!")

# %%

# determine resampling suggestion
med_dataset = fastMONAI.vision_all.MedDataset(
    img_list=df.seg_files.tolist(),
    dtype=fastMONAI.vision_all.MedMask
)
suggested_voxelsize, requires_resampling = med_dataset.suggestion()
largest_imagesize = med_dataset.get_largest_img_size(resample=suggested_voxelsize)

# %%
# k validation folds

marker_precisions = []
marker_recalls = []
precisions = []
recalls = []
fprs = []
tprs = []
fm_losses = []
calc_losses = []

kf = list(KFold(n_splits=k_folds, random_state=random_state, shuffle=True).split(df))

# %%
train_index = None
valid_index = None
for i in range(len(kf)):
    (train_index_i, valid_index_i) = kf[i]
    if df.iloc[valid_index_i].applymap(lambda cell: subject_name in str(cell)).any().any():
        (train_index, valid_index) = kf[i]
        break
if valid_index is None:
    print(f"ERROR: Subject {subject_name} not found")
else:
    print(f"Subject {subject_name} found in fold {i}")

# %%
dblock = fastMONAI.vision_all.MedDataBlock(
    blocks=(fastMONAI.vision_all.ImageBlock(cls=fastMONAI.vision_all.MedImage), fastMONAI.vision_all.MedMaskBlock),
    splitter=fastMONAI.vision_all.IndexSplitter(valid_index),
    get_x=fastMONAI.vision_all.ColReader('in_files'),
    get_y=fastMONAI.vision_all.ColReader('seg_files'),
    item_tfms=evaluation_augmentations,
    reorder=requires_resampling,
    resample=suggested_voxelsize
)

dls = fastMONAI.vision_all.DataLoaders.from_dblock(dblock, df, bs=batch_size)

n_input_channels = len(model_data[model_name]['in_files'][0].split(';'))
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

model_file = glob.glob(f"models/{model_name}-2*-*-{i}-best*")[0].replace('models/', '').replace('.pth', '')
learn = learn.load(model_file)
learn.model.cuda()

# Compute metrics on the entire training dataset
correct_markers = MarkersIdentified()

dblock_valid_eval = fastMONAI.vision_all.MedDataBlock(
    blocks=(fastMONAI.vision_all.ImageBlock(cls=fastMONAI.vision_all.MedImage), fastMONAI.vision_all.MedMaskBlock),
    splitter=fastMONAI.vision_all.IndexSplitter([]),
    get_x=fastMONAI.vision_all.ColReader('in_files'),
    get_y=fastMONAI.vision_all.ColReader('seg_files'),
    item_tfms=evaluation_augmentations,
    reorder=requires_resampling,
    resample=suggested_voxelsize
)
dls_valid_eval = fastMONAI.vision_all.DataLoaders.from_dblock(dblock_valid_eval, df.iloc[valid_index], bs=1, sampler=fastMONAI.vision_all.SequentialSampler)
for x, y in dls_valid_eval.train:
    pred = torch.argmax(learn.model(x), dim=1).unsqueeze(1).to(dtype=torch.float)
    correct_markers.accumulate(pred=pred.cpu(), targ=y.cpu())

marker_tps = correct_markers.overlap_count
marker_fps = correct_markers.pred_marker_count - correct_markers.overlap_count
marker_fns = correct_markers.targ_marker_count - correct_markers.overlap_count

marker_precision = (marker_tps / (marker_tps + marker_fps)) if (marker_tps + marker_fps) > 0 else 0
marker_recall = (marker_tps / (marker_tps + marker_fns))  if (marker_tps + marker_fns) > 0 else 0

loss, *metrics = learn.validate(ds_idx=0, dl=dls_valid_eval.train)

# %%
# get predictions
dls_valid_eval = fastMONAI.vision_all.DataLoaders.from_dblock(dblock_valid_eval, df.iloc[valid_index], bs=len(dls_valid_eval.train_ds), sampler=fastMONAI.vision_all.SequentialSampler)
subject_idx = dls_valid_eval.train.items.loc[valid_index_i][dls_valid_eval.train.items.loc[valid_index_i]['in_files'].str.contains(subject_name)].index[0]
subject_i = dls_valid_eval.train.items.index.get_loc(subject_idx)

# %%
x, y = list(dls_valid_eval.train)[0]
pred = learn.model(x).unsqueeze(1).to(dtype=torch.float)
pred_data = torch.Tensor(pred)
pred_empty = pred_data[subject_i,0,0,:,:,:]
pred_seed = pred_data[subject_i,0,1,:,:,:]
pred_calc = pred_data[subject_i,0,2,:,:,:]
pred_seg = torch.argmax(pred_data[subject_i,0,:,:,:], dim=0)

# %%
def pad_tensor(input_tensor, target_size, interpolation='trilinear'):
    # Calculate padding for each dimension
    pad_depth = max(0, target_size[0] - input_tensor.shape[0])
    pad_height = max(0, target_size[1] - input_tensor.shape[1])
    pad_width = max(0, target_size[2] - input_tensor.shape[2])

    # Apply padding
    padded_tensor = torch.nn.functional.pad(input_tensor, (pad_width//2, pad_width//2, pad_height//2, pad_height//2, pad_depth//2, pad_depth//2))

    if any(i < j for i, j in zip(target_size, padded_tensor.shape)):
        # Reshaping tensor shape for 5D input to F.interpolate()
        padded_tensor = padded_tensor.unsqueeze(0).unsqueeze(0).float() # convert to float

        # Use interpolation to resample
        if interpolation == 'nearest':
            resampled_tensor = torch.nn.functional.interpolate(padded_tensor, size=target_size, mode='nearest')
        else:
            resampled_tensor = torch.nn.functional.interpolate(padded_tensor, size=target_size, mode='trilinear', align_corners=False)

        # Convert back to original dtype, if needed
        resampled_tensor = resampled_tensor.to(input_tensor.dtype)

        # Remove added dimensions
        resampled_tensor = resampled_tensor.squeeze(0).squeeze(0)

        return resampled_tensor
    
    return padded_tensor

# %%
pred_empty = pad_tensor(pred_empty, (146, 160, 60))
pred_seed = pad_tensor(pred_seed, (146, 160, 60))
pred_calc = pad_tensor(pred_calc, (146, 160, 60))
pred_seg = pad_tensor(pred_seg.cpu(), (146, 160, 60), interpolation='nearest')

# %%
nii = nib.load(dls_valid_eval.train.items['in_files'].loc[subject_idx].split(';')[0])
nib.save(nib.Nifti1Image(dataobj=pred_seed.cpu().detach().numpy(), affine=nii.affine, header=nii.header), f"{subject_name}_pred_seed.nii")
nib.save(nib.Nifti1Image(dataobj=pred_calc.cpu().detach().numpy(), affine=nii.affine, header=nii.header), f"{subject_name}_pred_calc.nii")
nib.save(nib.Nifti1Image(dataobj=pred_empty.cpu().detach().numpy(), affine=nii.affine, header=nii.header), f"{subject_name}_pred_empty.nii")
nib.save(nib.Nifti1Image(dataobj=pred_seg.cpu().detach().numpy(), affine=nii.affine, header=nii.header), f"{subject_name}_pred_seg.nii")


# %%
