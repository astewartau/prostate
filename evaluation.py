# %%
import glob
import os
import json
import datetime
import time

import torch
import pandas as pd
import numpy as np
import nibabel as nib

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

#from fastai.vision.all import *
import fastMONAI.vision_all
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from scipy import interpolate

from useful_functions import *
from utils import *

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

colors = {
    'QSM' : '#a6cee3',
    'QSM-FMAP': '#1f78b4',
    'QSM-SWI' : '#b2df8a',
    'QSM-T1-R2s' : '#33a02c',
    'QSM-T1' : '#fb9a99',
    'FMAP': '#e31a1c',
    'SWI' : '#fdbf6f',
    'T1' : '#ff7f00',
    'R2s' : '#cab2d6',
    'GRE' : '#6a3d9a',
    'CT' : '#ffff99',
}

# %%
model_name = 'CT'
k_folds = 25
random_state = 42
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S')

batch_size = 6
training_epochs = 700
lr = 0.003
ce_loss_weights = torch.Tensor([1, 1, 1])
evaluation_augmentations = [
    fastMONAI.vision_all.PadOrCrop([80, 80, 80]),
    fastMONAI.vision_all.ZNormalization(),
]

# %%
plt.figure()
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
model_marker_precisions = {}
model_marker_recalls = {}
model_tps = {}
model_fps = {}
model_fps_markers_as_calcs = {}
model_fns = {}
model_in_files = {}
losses = {}

for model in model_data.keys():#['CT']:
    print(f"=== {model} ===")

    df = pd.DataFrame(model_data[model])
    infile_cols = [key for key in list(model_data[model].keys()) if key != 'seg_files']
    n_input_channels = len(infile_cols)
    print(f"infile_cols: {infile_cols}; n_input_channels: {n_input_channels}")

    # determine resampling suggestion
    if model == 'CT':
        med_dataset = fastMONAI.vision_all.MedDataset(
            img_list=df.seg_files.tolist(),
            dtype=fastMONAI.vision_all.MedMask
        )
        suggested_voxelsize, requires_resampling = med_dataset.suggestion()
        largest_imagesize = med_dataset.get_largest_img_size(resample=suggested_voxelsize)

    # k validation folds
    kf = KFold(n_splits=k_folds, random_state=random_state, shuffle=True)

    marker_precisions = []
    marker_recalls = []
    precisions = []
    recalls = []
    fprs = []
    tprs = []
    tps = []
    fps = []
    fps_markers_as_calcs = []
    fns = []
    fm_losses = []
    calc_losses = []
    in_files = []


    for i, (train_index, valid_index) in enumerate(kf.split(df)):

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

        model_file = glob.glob(f"models/{model}-2024*-*-{i}-best*")[0]
        print(f"Loading model: {model_file}")
        learn = learn.load(model_file.replace("models/", "").replace(".pth", ""))

        if torch.cuda.is_available():
            learn.model.cuda()

        # Compute metrics on the entire dataset
        correct_markers = MarkersIdentified()

        dblock_valid_eval = fastMONAI.vision_all.MedDataBlock(
            blocks=(fastMONAI.vision_all.ImageBlock(cls=fastMONAI.vision_all.MedImage), fastMONAI.vision_all.MedMaskBlock),
            splitter=fastMONAI.vision_all.IndexSplitter([]),
            get_x=fastMONAI.vision_all.ColReader(infile_cols),
            get_y=fastMONAI.vision_all.ColReader('seg_files'),
            item_tfms=evaluation_augmentations,
            reorder=requires_resampling,
            resample=suggested_voxelsize,
        )
        dls_valid_eval = fastMONAI.vision_all.DataLoaders.from_dblock(dblock_valid_eval, df.iloc[valid_index], bs=1, sampler=fastMONAI.vision_all.SequentialSampler)
        for i, (x, y) in enumerate(dls_valid_eval.train):
            print(f"{i}/{len(dls_valid_eval.train)}", end="; ")
            pred = torch.argmax(learn.model(x), dim=1).unsqueeze(1).to(dtype=torch.float)
            correct_markers.accumulate(pred=pred.cpu(), targ=y.cpu())

        marker_tps = correct_markers.overlap_count
        marker_fps = correct_markers.pred_marker_count - correct_markers.overlap_count
        marker_fns = correct_markers.targ_marker_count - correct_markers.overlap_count

        tps.append(marker_tps)
        fps.append(marker_fps)
        fps_markers_as_calcs.append(correct_markers.calcification_misclassifications)
        fns.append(marker_fns)


        marker_precision = (marker_tps / (marker_tps + marker_fps)) if (marker_tps + marker_fps) > 0 else 0
        marker_recall = (marker_tps / (marker_tps + marker_fns))  if (marker_tps + marker_fns) > 0 else 0
        marker_precisions.append(marker_precision)
        marker_recalls.append(marker_recall)

        loss, *metrics = learn.validate(ds_idx=0, dl=dls_valid_eval.train)
        fm_losses.append(metrics[0][0])
        calc_losses.append(metrics[0][1])

        print(f"Input file: {df.iloc[valid_index].iloc[0].iloc[0]}")
        print(f"Marker Precision: {marker_precision}")
        print(f"Marker Recall: {marker_recall}")
        print(f"True markers: {marker_tps}")
        print(f"False positive markers: {marker_fps} ({correct_markers.calcification_misclassifications} were known calcification)")
        print(f"False negative markers: {marker_fns}")
        print(f"Loss: {loss}")

        # get predictions
        dls_valid_eval = fastMONAI.vision_all.DataLoaders.from_dblock(dblock_valid_eval, df.iloc[valid_index], bs=len(dls_valid_eval.train_ds), sampler=fastMONAI.vision_all.SequentialSampler)
        valid_x, valid_y = dls_valid_eval.train.one_batch()

        def calc_stuff(x, y):
            pred = learn.model(x)[:,1,:,:,:].unsqueeze(1).cpu().detach().numpy()
            pred -= np.min(pred)
            pred /= np.max(pred)
            pred = pred.flatten()
            targ = (y.cpu() == 1).to(dtype=torch.int).detach().numpy().flatten()

            # calculate AUC
            sample_weight = compute_sample_weight(class_weight="balanced", y=targ, indices=None)
            fpr, tpr, thresholds = roc_curve(targ, pred, sample_weight=sample_weight)
            roc_auc = auc(fpr, tpr)

            # calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(targ, pred)
            average_precision = average_precision_score(targ, pred)

            return fpr, tpr, precision, recall, thresholds, average_precision, roc_auc

        fpr, tpr, precision, recall, thresholds, average_tpr, roc_auc = calc_stuff(valid_x, valid_y)

        in_files.append(df.iloc[valid_index].iloc[0].iloc[0])
        precisions.append(precision)
        recalls.append(recall)
        fprs.append(fpr)
        tprs.append(tpr)

    model_marker_precisions[model] = [np.mean(marker_precisions), np.std(marker_precisions)]
    model_marker_recalls[model] = [np.mean(marker_recalls), np.std(marker_recalls)]
    losses[model] = [np.mean(fm_losses), np.std(fm_losses), np.mean(calc_losses), np.std(calc_losses)]

    model_tps[model] = tps
    model_fps[model] = fps
    model_fps_markers_as_calcs[model] = fps_markers_as_calcs
    model_fns[model] = fns
    model_in_files[model] = in_files

    # define a common set of FPR values
    common_fpr = np.linspace(0, 1, 100)

    # initialize an empty array to hold the interpolated TPRs
    interpolated_tprs = []

    for tpr, fpr in zip(tprs, fprs):
        # interpolate the TPR values to the common FPR values
        interpolated_tpr = np.interp(common_fpr, fpr, tpr)
        
        # store the interpolated TPR
        interpolated_tprs.append(interpolated_tpr)

    # calculate the average TPR at each common FPR value
    average_tpr = np.mean(interpolated_tprs, axis=0)

    roc_auc = auc(common_fpr, average_tpr)

    # plot the average precision-recall curve
    plt.plot(common_fpr, average_tpr, color=colors[model], label=f'{model} (AUC = {round(roc_auc, 2)})')
    

    del learn, dls, dblock_valid_eval, dls_valid_eval, loss, metrics, valid_x, valid_y, correct_markers

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic (ROC) Curves\n(Average over {k_folds}-fold cross-validation)')
plt.legend(loc="lower right")
plt.savefig("roc-curves-new3.png", dpi=400)
plt.show()

# %%
for model, in_file in model_in_files.items():
    print(f"{model.rjust(10)}", in_file)

# %%
for model, tps in model_tps.items():
    print(f"{model.rjust(10)}", tps)

# %%
for model, fns in model_fns.items():
    print(f"{model.rjust(10)}", fns)

# %%
for model, fps in model_fps.items():
    print(f"{model.rjust(10)}", fps)

# %%
for model, fps_markers_as_calcs in model_fps_markers_as_calcs.items():
    print(f"{model.rjust(10)}", fps_markers_as_calcs)

# %%
# Assuming model_tps, model_fps, model_fns, and model_fps_markers_as_calcs are dictionaries with lists of values
summary = {}

for model in model_tps.keys():
    summary[model] = {
        "Total markers": np.sum(model_tps[model]) + np.sum(model_fns[model]),
        "Recalled markers": np.sum(model_tps[model]),
        "Missed markers": np.sum(model_fns[model]),
        "Incorrect/extra markers": np.sum(model_fps[model]),
        "Calcifications confused as FMs": np.sum(model_fps_markers_as_calcs[model])
    }

df_summary = pd.DataFrame.from_dict(summary, orient='index')
df_summary.sort_values(list(df_summary.columns), ascending=False)

# %%
import pandas as pd

# Initialize lists to store marker counts for each file and model
results = []

plt.figure()
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
model_marker_precisions = {}
model_marker_recalls = {}
losses = {}

for model in model_data.keys():
    print(f"=== {model} ===")

    df = pd.DataFrame(model_data[model])
    infile_cols = [key for key in list(model_data[model].keys()) if key != 'seg_files']
    n_input_channels = len(infile_cols)
    print(f"infile_cols: {infile_cols}; n_input_channels: {n_input_channels}")

    # Determine resampling suggestion
    if model == 'CT':
        med_dataset = fastMONAI.vision_all.MedDataset(
            img_list=df.seg_files.tolist(),
            dtype=fastMONAI.vision_all.MedMask
        )
        suggested_voxelsize, requires_resampling = med_dataset.suggestion()
        largest_imagesize = med_dataset.get_largest_img_size(resample=suggested_voxelsize)

    # k validation folds
    kf = KFold(n_splits=k_folds, random_state=random_state, shuffle=True)

    marker_precisions = []
    marker_recalls = []
    precisions = []
    recalls = []
    fprs = []
    tprs = []
    fm_losses = []
    calc_losses = []

    for i, (train_index, valid_index) in enumerate(kf.split(df)):

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
            metrics=[fastMONAI.vision_all.multi_dice_score, MarkersIdentified(), SuperfluousMarkers()]
        )

        model_file = glob.glob(f"models/{model}-2024*-*-{i}-best*")[0]
        print(model_file)
        learn = learn.load(model_file.replace("models/", "").replace(".pth", ""))

        if torch.cuda.is_available():
            learn.model.cuda()

        # Compute metrics on the entire dataset
        correct_markers = MarkersIdentified()

        dblock_valid_eval = fastMONAI.vision_all.MedDataBlock(
            blocks=(fastMONAI.vision_all.ImageBlock(cls=fastMONAI.vision_all.MedImage), fastMONAI.vision_all.MedMaskBlock),
            splitter=fastMONAI.vision_all.IndexSplitter([]),
            get_x=fastMONAI.vision_all.ColReader(infile_cols),
            get_y=fastMONAI.vision_all.ColReader('seg_files'),
            item_tfms=evaluation_augmentations,
            reorder=requires_resampling,
            resample=suggested_voxelsize,
        )
        dls_valid_eval = fastMONAI.vision_all.DataLoaders.from_dblock(dblock_valid_eval, df.iloc[valid_index], bs=1, sampler=fastMONAI.vision_all.SequentialSampler)
        for j, (x, y) in enumerate(dls_valid_eval.train):
            print(f"{j}/{len(dls_valid_eval.train)}", end="; ")
            pred = torch.argmax(learn.model(x), dim=1).unsqueeze(1).to(dtype=torch.float)
            correct_markers.accumulate(pred=pred.cpu(), targ=y.cpu())

            # Store marker counts
            results.append({
                'File': df.iloc[valid_index].iloc[j].iloc[0],
                'Model': model,
                '# Target Markers': correct_markers.targ_marker_count,
                '# Predicted Markers': correct_markers.pred_marker_count,
                '# Overlapping': correct_markers.overlap_count
            })

        marker_tps = correct_markers.overlap_count
        marker_fps = correct_markers.pred_marker_count - correct_markers.overlap_count
        marker_fns = correct_markers.targ_marker_count - correct_markers.overlap_count

        marker_precision = (marker_tps / (marker_tps + marker_fps)) if (marker_tps + marker_fps) > 0 else 0
        marker_recall = (marker_tps / (marker_tps + marker_fns))  if (marker_tps + marker_fns) > 0 else 0
        marker_precisions.append(marker_precision)
        marker_recalls.append(marker_recall)

        loss, *metrics = learn.validate(ds_idx=0, dl=dls_valid_eval.train)
        fm_losses.append(metrics[0][0])
        calc_losses.append(metrics[0][1])

        print(f"Input file: {df.iloc[valid_index].iloc[0].iloc[0]}")
        print(f"Marker Precision: {marker_precision}")
        print(f"Marker Recall: {marker_recall}")
        print(f"Loss: {loss}")

        # Get predictions
        dls_valid_eval = fastMONAI.vision_all.DataLoaders.from_dblock(dblock_valid_eval, df.iloc[valid_index], bs=len(dls_valid_eval.train_ds), sampler=fastMONAI.vision_all.SequentialSampler)
        valid_x, valid_y = dls_valid_eval.train.one_batch()

        def calc_stuff(x, y):
            pred = learn.model(x)[:,1,:,:,:].unsqueeze(1).cpu().detach().numpy()
            pred -= np.min(pred)
            pred /= np.max(pred)
            pred = pred.flatten()
            targ = (y.cpu() == 1).to(dtype=torch.int).detach().numpy().flatten()

            # Calculate AUC
            sample_weight = compute_sample_weight(class_weight="balanced", y=targ, indices=None)
            fpr, tpr, thresholds = roc_curve(targ, pred, sample_weight=sample_weight)
            roc_auc = auc(fpr, tpr)

            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(targ, pred)
            average_precision = average_precision_score(targ, pred)

            return fpr, tpr, precision, recall, thresholds, average_precision, roc_auc

        fpr, tpr, precision, recall, thresholds, average_tpr, roc_auc = calc_stuff(valid_x, valid_y)

        precisions.append(precision)
        recalls.append(recall)
        fprs.append(fpr)
        tprs.append(tpr)

    model_marker_precisions[model] = [np.mean(marker_precisions), np.std(marker_precisions)]
    model_marker_recalls[model] = [np.mean(marker_recalls), np.std(marker_recalls)]
    losses[model] = [np.mean(fm_losses), np.std(fm_losses), np.mean(calc_losses), np.std(calc_losses)]

    # Define a common set of FPR values
    common_fpr = np.linspace(0, 1, 100)

    # Initialize an empty array to hold the interpolated TPRs
    interpolated_tprs = []

    for tpr, fpr in zip(tprs, fprs):
        # Interpolate the TPR values to the common FPR values
        interpolated_tpr = np.interp(common_fpr, fpr, tpr)
        
        # Store the interpolated TPR
        interpolated_tprs.append(interpolated_tpr)

    # Calculate the average TPR at each common FPR value
    average_tpr = np.mean(interpolated_tprs, axis=0)

    roc_auc = auc(common_fpr, average_tpr)

    # Plot the average precision-recall curve
    plt.plot(common_fpr, average_tpr, color=colors[model], label=f'{model} (AUC = {round(roc_auc, 2)})')
    

    del learn, dls, dblock_valid_eval, dls_valid_eval, loss, metrics, valid_x, valid_y, correct_markers

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic (ROC) Curves\n(Average over {k_folds}-fold cross-validation)')
plt.legend(loc="lower right")
plt.savefig("roc-curves-new3.png", dpi=400)
plt.show()

print(model_marker_precisions)
print(model_marker_recalls)
print(losses)

# Create and display the DataFrame with the results
df_results = pd.DataFrame(results)
print(df_results)

# %%
print(df_results)

# %%
plt.figure()
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

total_samples = 0
positive_samples = 0

model_marker_precisions = {}
model_marker_recalls = {}
losses = {}
for model in model_data.keys():
    print(f"=== {model} ===")

    df = pd.DataFrame(model_data[model])
    
    infile_cols = [key for key in list(model_data[model].keys()) if key != 'seg_files']
    n_input_channels = len(infile_cols)
    print(f"infile_cols: {infile_cols}; n_input_channels: {n_input_channels}")

    if model == 'CT':
        # determine resampling suggestion
        med_dataset = fastMONAI.vision_all.MedDataset(
            img_list=df.seg_files.tolist(),
            dtype=fastMONAI.vision_all.MedMask
        )
        suggested_voxelsize, requires_resampling = med_dataset.suggestion()
        largest_imagesize = med_dataset.get_largest_img_size(resample=suggested_voxelsize)

    # k validation folds
    kf = KFold(n_splits=k_folds, random_state=random_state, shuffle=True)

    marker_precisions = []
    marker_recalls = []
    precisions = []
    recalls = []
    fprs = []
    tprs = []
    fm_losses = []
    calc_losses = []

    for i, (train_index, valid_index) in enumerate(kf.split(df)):
        if model == 'QSM':
            y_values = df.iloc[valid_index]['seg_files'].tolist()
            for y_file in y_values:
                y_data = np.array(nib.load(y_file).get_fdata() == 1, dtype=int)
                total_samples += y_data.size
                positive_samples += int(y_data.sum())

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
                weight=ce_loss_weights
            ),
            opt_func=fastMONAI.vision_all.ranger,
            metrics=[fastMONAI.vision_all.multi_dice_score, MarkersIdentified(), SuperfluousMarkers()]#.to_fp16()
        )

        model_file = glob.glob(f"models/{model}-2024*-*-{i}-best*")[0]
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
        marker_precisions.append(marker_precision)
        marker_recalls.append(marker_recall)

        loss, *metrics = learn.validate(ds_idx=0, dl=dls_valid_eval.train)
        fm_losses.append(metrics[0][0])
        calc_losses.append(metrics[0][1])

        print(f"Marker Precision: {marker_precision}")
        print(f"Marker Recall: {marker_recall}")
        print(f"Loss: {loss}")

        # get predictions
        dls_valid_eval = fastMONAI.vision_all.DataLoaders.from_dblock(dblock_valid_eval, df.iloc[valid_index], bs=len(dls_valid_eval.train_ds), sampler=fastMONAI.vision_all.SequentialSampler)
        valid_x, valid_y = dls_valid_eval.train.one_batch()

        def calc_stuff(x, y):
            pred = learn.model(x)[:,1,:,:,:].unsqueeze(1).cpu().detach().numpy()
            pred -= np.min(pred)
            pred /= np.max(pred)
            pred = pred.flatten()
            targ = (y.cpu() == 1).to(dtype=torch.int).detach().numpy().flatten()

            # calculate AUC
            sample_weight = compute_sample_weight(class_weight="balanced", y=targ, indices=None)
            fpr, tpr, thresholds = roc_curve(targ, pred, sample_weight=sample_weight)
            roc_auc = auc(fpr, tpr)

            # calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(targ, pred)
            average_precision = average_precision_score(targ, pred)

            return fpr, tpr, precision, recall, thresholds, average_precision, roc_auc

        fpr, tpr, precision, recall, thresholds, average_tpr, roc_auc = calc_stuff(valid_x, valid_y)

        precisions.append(precision)
        recalls.append(recall)
        fprs.append(fpr)
        tprs.append(tpr)

    if model == 'QSM':
        positive_ratio = positive_samples / total_samples
        #plt.hlines(positive_ratio, xmin=0, xmax=1, color='navy', linestyle='--')

    model_marker_precisions[model] = [np.mean(marker_precisions), np.std(marker_precisions)]
    model_marker_recalls[model] = [np.mean(marker_recalls), np.std(marker_recalls)]
    losses[model] = [np.mean(fm_losses), np.std(fm_losses), np.mean(calc_losses), np.std(calc_losses)]
    
    # define a common set of recall values
    common_recall = np.linspace(0, 1, 100)

    # initialize an empty array to hold the interpolated precisions
    interpolated_precisions = []

    for precision, recall in zip(precisions, recalls):
        # reverse the arrays because recall should be non-decreasing for interpolation
        precision = precision[::-1]
        recall = recall[::-1]

        # interpolate the precision values to the common recall values
        interpolated_precision = np.interp(common_recall, recall, precision)
        
        # store the interpolated precision
        interpolated_precisions.append(interpolated_precision)

    # calculate the average precision at each common recall value
    average_precision = np.mean(interpolated_precisions, axis=0)

    prc_auc = auc(common_recall, average_precision)

    # plot the average precision-recall curve
    plt.plot(common_recall, average_precision, color=colors[model], label=f'{model} (AUC = {round(prc_auc, 2)})')

    if model == 'QSM':
        positive_ratio = np.mean([y.cpu().mean() for _, y in dls_valid_eval.train])  # assuming y contains binary labels with 1s for positive samples.
        plt.hlines(positive_ratio, xmin=0, xmax=1, color='navy', linestyle='--')

    del learn, dls, dblock_valid_eval, dls_valid_eval, loss, metrics, valid_x, valid_y, correct_markers

print(model_marker_precisions)
print(model_marker_recalls)
print(losses)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curves\n(Average over {k_folds}-fold cross-validation)')
#plt.legend(loc="lower right")
plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
plt.tight_layout()  # to ensure that all the elements fit into the figure area
plt.savefig("poc-curves-marker.png", dpi=400)
plt.show()

# %%
# Compute average fpr, tpr, and auc for each model category and plot ROC curve
for model in model_data.keys():
    avg_fpr = np.mean(model_fprs[model], axis=0)
    avg_tpr = np.mean(model_tprs[model], axis=0)
    avg_auc = np.mean(model_aucs[model])

    plt.plot(avg_fpr, avg_tpr, color=colors[model], label=f'{model} (AUC = {round(avg_auc, 2)})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("roc-curve-new.png", dpi=400)
plt.show()

print(model_marker_precisions)
print(model_marker_recalls)
print(losses)

# %%
print(model_fprs.keys())
for key in model_fprs.keys():
    print(type(model_fprs[key]))
    print(len(model_fprs[key]))
    for element in model_fprs[key]:
        print(type(element))
        print(len(element))
x = np.array(model_fprs[model])

# %%
print(model_marker_precisions)
print(model_marker_recalls)
print(losses)

# %%
pd.DataFrame(model_marker_precisions)

# %%
pd.DataFrame(model_marker_recalls)

# %%
# Placeholder for legend labels and their corresponding AUCs
legend_data = []

for model in ['CT']:# model_data.keys():
    mean_fpr = np.linspace(0, 1, 100)
    tpr_interp = []
    
    for i in range(len(model_fprs[model])):
        fpr = model_fprs[model][i]
        tpr = model_tprs[model][i]
        tpr_interp.append(np.interp(mean_fpr, fpr, tpr))
    
    tpr_array = np.array(tpr_interp)
    avg_tpr = tpr_array.mean(axis=0)
    avg_auc = np.mean(model_aucs[model])

    # Store label and AUC for later sorting
    legend_data.append((model, avg_auc))

    plt.plot(mean_fpr, avg_tpr, color=colors[model], label=f'{model} (AUC = {avg_auc:.2f})')

# Sort the legend data based on AUC in descending order
legend_data.sort(key=lambda x: x[1], reverse=True)

# Create custom legend
custom_legend = [plt.Line2D([0], [0], color=colors[label], lw=4) for label, _ in legend_data]
plt.legend(custom_legend, [f'{label} (AUC = {auc:.2f})' for label, auc in legend_data], loc='lower right')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves\n(average of old models on new data)')
plt.show()

# %%
plt.figure()
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

total_samples = 0
positive_samples = 0


model_marker_precisions = {}
model_marker_recalls = {}
losses = {}
for model in model_data.keys():
    print(f"=== {model} ===")

    df = pd.DataFrame(model_data[model])
    infile_cols = [key for key in list(model_data[model].keys()) if key != 'seg_files']
    n_input_channels = len(infile_cols)
    print(f"infile_cols: {infile_cols}; n_input_channels: {n_input_channels}")

    if model == 'CT':
        # determine resampling suggestion
        med_dataset = fastMONAI.vision_all.MedDataset(
            img_list=df.seg_files.tolist(),
            dtype=fastMONAI.vision_all.MedMask
        )
        suggested_voxelsize, requires_resampling = med_dataset.suggestion()
        largest_imagesize = med_dataset.get_largest_img_size(resample=suggested_voxelsize)

    # k validation folds
    kf = KFold(n_splits=k_folds, random_state=random_state, shuffle=True)

    marker_precisions = []
    marker_recalls = []
    precisions = []
    recalls = []
    fprs = []
    tprs = []
    fm_losses = []
    calc_losses = []

    for i, (train_index, valid_index) in enumerate(kf.split(df)):
        if model == 'QSM':
            y_values = df.iloc[valid_index]['seg_files'].tolist()
            for y_file in y_values:
                y_data = np.array(nib.load(y_file).get_fdata() == 1, dtype=int)
                total_samples += y_data.size
                positive_samples += int(y_data.sum())

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
                weight=ce_loss_weights
            ),
            opt_func=fastMONAI.vision_all.ranger,
            metrics=[fastMONAI.vision_all.multi_dice_score, MarkersIdentified(), SuperfluousMarkers()]#.to_fp16()
        )

        model_file = glob.glob(f"models/{model}-2024*-*-{i}-best*")[0]
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
        marker_precisions.append(marker_precision)
        marker_recalls.append(marker_recall)

        loss, *metrics = learn.validate(ds_idx=0, dl=dls_valid_eval.train)
        fm_losses.append(metrics[0][0])
        calc_losses.append(metrics[0][1])

        print(f"Marker Precision: {marker_precision}")
        print(f"Marker Recall: {marker_recall}")
        print(f"Loss: {loss}")

        # get predictions
        dls_valid_eval = fastMONAI.vision_all.DataLoaders.from_dblock(dblock_valid_eval, df.iloc[valid_index], bs=len(dls_valid_eval.train_ds), sampler=fastMONAI.vision_all.SequentialSampler)
        valid_x, valid_y = dls_valid_eval.train.one_batch()

        def calc_stuff(x, y):
            pred = learn.model(x)[:,1,:,:,:].unsqueeze(1).cpu().detach().numpy()
            pred -= np.min(pred)
            pred /= np.max(pred)
            pred = pred.flatten()
            targ = (y.cpu() == 1).to(dtype=torch.int).detach().numpy().flatten()

            # calculate AUC
            sample_weight = compute_sample_weight(class_weight="balanced", y=targ, indices=None)
            fpr, tpr, thresholds = roc_curve(targ, pred, sample_weight=sample_weight)
            roc_auc = auc(fpr, tpr)

            # calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(targ, pred)
            average_precision = average_precision_score(targ, pred)

            return fpr, tpr, precision, recall, thresholds, average_precision, roc_auc

        fpr, tpr, precision, recall, thresholds, average_tpr, roc_auc = calc_stuff(valid_x, valid_y)

        precisions.append(precision)
        recalls.append(recall)
        fprs.append(fpr)
        tprs.append(tpr)

    if model == 'QSM':
        positive_ratio = positive_samples / total_samples
        #plt.hlines(positive_ratio, xmin=0, xmax=1, color='navy', linestyle='--')

    model_marker_precisions[model] = [np.mean(marker_precisions), np.std(marker_precisions)]
    model_marker_recalls[model] = [np.mean(marker_recalls), np.std(marker_recalls)]
    losses[model] = [np.mean(fm_losses), np.std(fm_losses), np.mean(calc_losses), np.std(calc_losses)]
    
    # define a common set of recall values
    common_recall = np.linspace(0, 1, 100)

    # initialize an empty array to hold the interpolated precisions
    interpolated_precisions = []

    for precision, recall in zip(precisions, recalls):
        # reverse the arrays because recall should be non-decreasing for interpolation
        precision = precision[::-1]
        recall = recall[::-1]

        # interpolate the precision values to the common recall values
        interpolated_precision = np.interp(common_recall, recall, precision)
        
        # store the interpolated precision
        interpolated_precisions.append(interpolated_precision)

    # calculate the average precision at each common recall value
    average_precision = np.mean(interpolated_precisions, axis=0)

    prc_auc = auc(common_recall, average_precision)

    # plot the average precision-recall curve
    plt.plot(common_recall, average_precision, color=colors[model], label=f'{model} (AUC = {round(prc_auc, 2)})')

    if model == 'QSM':
        positive_ratio = np.mean([y.mean() for _, y in dls_valid_eval.train])  # assuming y contains binary labels with 1s for positive samples.
        plt.hlines(positive_ratio, xmin=0, xmax=1, color='navy', linestyle='--')

    del learn, dls, dblock_valid_eval, dls_valid_eval, loss, metrics, valid_x, valid_y, correct_markers

print(model_marker_precisions)
print(model_marker_recalls)
print(losses)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curves\n(Average over {k_folds}-fold cross-validation)')
#plt.legend(loc="lower right")
plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
plt.tight_layout()  # to ensure that all the elements fit into the figure area
plt.savefig("poc-curves-marker.png", dpi=400)
plt.show()

# %%
import seaborn as sns

# Convert the dictionary to a pandas dataframe
df = pd.DataFrame(model_marker_precisions).transpose().reset_index()
df.columns = ['Model', 'Mean', 'Std']

# Create a barplot using seaborn
plt.figure(figsize=(10,6))
sns.barplot(data=df, x='Model', y='Mean', yerr=df['Std'], capsize=.2)

plt.title('Precision Across Models')
plt.ylabel('Precision')
plt.xlabel('Models')
plt.show()

# %%
# Convert the dictionary to a pandas dataframe
df = pd.DataFrame(model_marker_recalls).transpose().reset_index()
df.columns = ['Model', 'Mean', 'Std']

# Create a barplot using seaborn
plt.figure(figsize=(10,6))
sns.barplot(data=df, x='Model', y='Mean', yerr=df['Std'], capsize=.2)

plt.title('Recall Across Models')
plt.ylabel('Recall')
plt.xlabel('Models')
plt.show()

# %%
# Convert the dictionary to a pandas dataframe
prec_df = pd.DataFrame(model_marker_precisions).transpose().reset_index()
rec_df = pd.DataFrame(model_marker_recalls).transpose().reset_index()

# Add Metric column to differentiate between Precision and Recall
prec_df['Metric'] = 'Precision'
rec_df['Metric'] = 'Recall'

prec_df.columns = ['Model', 'Mean', 'Std', 'Metric']
rec_df.columns = ['Model', 'Mean', 'Std', 'Metric']

# Combine both dataframes
df = pd.concat([prec_df, rec_df])

# %%
# Convert 'Mean' and 'Std' to the desired string format
df['Mean +/- Std'] = df.apply(lambda row: f'{round(row["Mean"], 2)} +/- {round(row["Std"], 2)}', axis=1)

# Pivot the table to get the desired format
df_pivot = df.pivot(index='Model', columns='Metric', values='Mean +/- Std').reset_index()

# Rename the columns to 'Recall' and 'Precision'
df_pivot.columns = ['Model', 'Precision', 'Recall']
df_pivot = df_pivot.sort_values('Precision', ascending=False)

# %%
df_pivot

# %%
# Create a grouped barplot using seaborn
plt.figure(figsize=(12,8))
barplot = sns.barplot(data=df, x='Model', y='Mean', hue='Metric', capsize=.2)

# Add error bars manually
for i, model in enumerate(df['Model'].unique()):
    model_data = df[df['Model'] == model]
    precision_data = model_data[model_data['Metric'] == 'Precision']
    recall_data = model_data[model_data['Metric'] == 'Recall']
    barplot.errorbar(x=i - 0.2, y=precision_data['Mean'].values[0], yerr=precision_data['Std'].values[0], color='black', capsize=3)
    barplot.errorbar(x=i + 0.2, y=recall_data['Mean'].values[0], yerr=recall_data['Std'].values[0], color='black', capsize=3)

plt.ylim(0, 1.05)
plt.title('Marker-level Precision and Recall across Models')
plt.ylabel('Score')
plt.xlabel('Models')
plt.show()

# %%
losses

# %%
# Initialize lists to hold the data
model_list = []
region_list = []
mean_list = []
std_list = []

# Loop over the dictionary
for model, values in losses.items():
    # Add 'FM' region values
    model_list.append(model)
    region_list.append('FM')
    mean_list.append(values[0])
    std_list.append(values[1])
    
    # Add 'Calc' region values
    model_list.append(model)
    region_list.append('Calc')
    mean_list.append(values[2])
    std_list.append(values[3])

# Create the DataFrame
losses_df = pd.DataFrame({
    'Model': model_list,
    'Region': region_list,
    'Mean': mean_list,
    'Std': std_list
})
losses_df

# %%
# Create a grouped barplot using seaborn
plt.figure(figsize=(12,8))
barplot = sns.barplot(data=losses_df, x='Model', y='Mean', hue='Region', capsize=.2)

# Add error bars manually
for i, model in enumerate(losses_df['Model'].unique()):
    for j, region in enumerate(losses_df['Region'].unique()):
        model_data = losses_df[losses_df['Model'] == model]
        model_data = model_data[model_data['Region'] == region]
        barplot.errorbar(x=i - 0.2 + 0.4*j, y=model_data['Mean'].values[0], yerr=model_data['Std'].values[0], color='black', capsize=3)

plt.ylim(0, 1.0)
plt.title('Dice scores across models and regions')
plt.ylabel('Dice score')
plt.xlabel('Models')
plt.show()

# %%
pred_calc_files = sorted(sum((glob.glob(os.path.join(session_dir, "extra_data", "*pred_calc.nii*")) for session_dir in session_dirs), []))
pred_seed_files = sorted(sum((glob.glob(os.path.join(session_dir, "extra_data", "*pred_seed.nii*")) for session_dir in session_dirs), []))
pred_empty_files = sorted(sum((glob.glob(os.path.join(session_dir, "extra_data", "*pred_empty.nii*")) for session_dir in session_dirs), []))
pred_seg_files = sorted(sum((glob.glob(os.path.join(session_dir, "extra_data", "*pred_seg.nii*")) for session_dir in session_dirs), []))

assert(len(qsm_files) == len(pred_calc_files))
assert(len(qsm_files) == len(pred_seed_files))
assert(len(qsm_files) == len(pred_empty_files))
assert(len(qsm_files) == len(pred_seg_files))

# %%
input_images = qsm_files
#input_segmentations = ct_seg_clean_files
#raw_segmentations = ct_seg_raw_files
input_segmentations = gre_seg_clean_files
raw_segmentations = gre_seg_raw_files
#clip_data = (0, 100)
clip_data = None
#vrange = (0, 1000) # T1w
#vrange = (0, 0.75) # SWI
vrange = (-0.5, +0.5) # QSM
#vrange = (0, 800) # GRE
crop_size = 20
half_cropsize = crop_size // 2

regions_vals = []
regions_masks = []
regions_pred_segs = []
regions_subjects = []
regions_pred_seeds = []
regions_pred_calcs = []

# %%

for i in range(len(input_segmentations)):
    # load images
    subject = input_segmentations[i].split(os.sep)[1]
    seg = nib.load(input_segmentations[i]).get_fdata()
    input_data = nib.load(input_images[i]).get_fdata()
    pred_seg_data = nib.load(pred_seg_files[i]).get_fdata()
    pred_seed_data = nib.load(pred_seed_files[i]).get_fdata()
    pred_calc_data = nib.load(pred_calc_files[i]).get_fdata()

    if clip_data is not None:
        input_data[input_data == np.inf] = clip_data[1]
        input_data[input_data < 0] = clip_data[0]
        input_data[input_data > 100] = clip_data[1]
        input_data[np.isnan(input_data)] = 0

    # get regions
    centroids = get_centroids(mask=seg == SegTypeClean.GOLD_SEED.value)

    for j in range(len(centroids)):
        seg_submask = seg[
            centroids[j][0]-half_cropsize:centroids[j][0]+half_cropsize,
            centroids[j][1]-half_cropsize:centroids[j][1]+half_cropsize,
            centroids[j][2]-half_cropsize:centroids[j][2]+half_cropsize
        ]
        pred_seg_submask = pred_seg_data[
            centroids[j][0]-half_cropsize:centroids[j][0]+half_cropsize,
            centroids[j][1]-half_cropsize:centroids[j][1]+half_cropsize,
            centroids[j][2]-half_cropsize:centroids[j][2]+half_cropsize
        ]
        pred_seed_submask = pred_seed_data[
            centroids[j][0]-half_cropsize:centroids[j][0]+half_cropsize,
            centroids[j][1]-half_cropsize:centroids[j][1]+half_cropsize,
            centroids[j][2]-half_cropsize:centroids[j][2]+half_cropsize
        ]
        pred_calc_submask = pred_calc_data[
            centroids[j][0]-half_cropsize:centroids[j][0]+half_cropsize,
            centroids[j][1]-half_cropsize:centroids[j][1]+half_cropsize,
            centroids[j][2]-half_cropsize:centroids[j][2]+half_cropsize
        ]
        subvals = input_data[
            centroids[j][0]-half_cropsize:centroids[j][0]+half_cropsize,
            centroids[j][1]-half_cropsize:centroids[j][1]+half_cropsize,
            centroids[j][2]-half_cropsize:centroids[j][2]+half_cropsize
        ]
        regions_vals.append(subvals)
        regions_masks.append(seg_submask)
        regions_pred_segs.append(pred_seg_submask)
        regions_pred_seeds.append(pred_seed_submask)
        regions_pred_calcs.append(pred_calc_submask)
        regions_subjects.append(subject)

# %%


print("Creating figure")
fig, axes = plt.subplots(ncols=4, nrows=len(regions_vals), figsize=(10, 180))

for ax in axes.flat:
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

for i in range(len(regions_vals)):
    try:
        axes[i,0].imshow(regions_vals[i][regions_vals[i].shape[0]//2,:,:], cmap='gray', vmin=vrange[0], vmax=vrange[1], interpolation='nearest')
        
        axes[i,1].imshow(regions_vals[i][regions_vals[i].shape[0]//2,:,:], cmap='gray', vmin=vrange[0], vmax=vrange[1], interpolation='nearest')
        axes[i,1].imshow(regions_masks[i][regions_vals[i].shape[0]//2,:,:], cmap='tab10', alpha=np.array(regions_masks[i][regions_vals[i].shape[0]//2,:,:], dtype=float), vmin=1, vmax=9)

        axes[i,2].imshow(regions_vals[i][regions_vals[i].shape[0]//2,:,:], cmap='gray', vmin=vrange[0], vmax=vrange[1], interpolation='nearest')
        axes[i,2].imshow(regions_pred_segs[i][regions_vals[i].shape[0]//2,:,:], cmap='tab10', alpha=np.array(regions_pred_segs[i][regions_vals[i].shape[0]//2,:,:], dtype=float), vmin=1, vmax=9)

        axes[i,3].imshow(regions_vals[i][regions_vals[i].shape[0]//2,:,:], cmap='gray', vmin=vrange[0], vmax=vrange[1], interpolation='nearest')
        seed_alphamap = regions_pred_seeds[i][regions_vals[i].shape[0]//2,:,:] > 0
        calc_alphamap = regions_pred_calcs[i][regions_vals[i].shape[0]//2,:,:] > 0
        calc_alphamap = np.logical_and(calc_alphamap, calc_alphamap > seed_alphamap)
        calc_alphamap = np.logical_and(calc_alphamap, regions_pred_segs[i][regions_vals[i].shape[0]//2,:,:] != 2)
        axes[i,3].imshow(regions_pred_seeds[i][regions_vals[i].shape[0]//2,:,:], cmap='winter', alpha=np.array(seed_alphamap, dtype=float), vmin=0.01, vmax=15)
        axes[i,3].imshow(regions_pred_calcs[i][regions_vals[i].shape[0]//2,:,:], cmap='autumn', alpha=np.array(calc_alphamap, dtype=float), vmin=0.01, vmax=15)
    except:
        continue
    axes[i,0].set_ylabel(regions_subjects[i], rotation=0, fontsize=12, labelpad=55)

#print("Saving figure")
plt.savefig("seeds_qsm_preds.png", bbox_inches='tight', dpi=200)

print("Displaying figure")
plt.show()
plt.close()

# %%



