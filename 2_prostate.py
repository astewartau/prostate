# %%
import torch

# %%
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU is available. Fastai will use the GPU.")
else:
    device = torch.device("cpu")
    print("GPU is NOT available. Fastai will use the CPU.")

# %%
import glob
import os
import json
import time
import datetime
import sys

import pandas as pd
import numpy as np
import nibabel as nib

import random
import fastai
import fastcore.transform
import fastMONAI.vision_all
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.transforms import MapTransform
from sklearn.model_selection import KFold

import scipy.ndimage 
from sklearn.model_selection import train_test_split
from skimage.measure import label, regionprops

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
session_dirs

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

# %% [markdown]
# # Parameters

# %%
model_data = {
    'CT' : { 'in_files' : [f"{ct_files[i]}" for i in range(len(ct_files))], 'seg_files': ct_seg_clean_files },
    'QSM-T1-T2s' : { 'in_files' : [f"{qsm_files[i]};{t1_resampled_files[i]};{t2s_files[i]}" for i in range(len(qsm_files))], 'seg_files': gre_seg_clean_files },
    'QSM-T1-R2s' : { 'in_files' : [f"{qsm_files[i]};{t1_resampled_files[i]};{r2s_files[i]}" for i in range(len(qsm_files))], 'seg_files': gre_seg_clean_files },
    'QSM-T1' : { 'in_files' : [f"{qsm_files[i]};{t1_resampled_files[i]}" for i in range(len(qsm_files))], 'seg_files': gre_seg_clean_files },
    'QSM' : { 'in_files' : [f"{qsm_files[i]}" for i in range(len(qsm_files))], 'seg_files': gre_seg_clean_files },
    'SWI' : { 'in_files' : [f"{swi_files[i]}" for i in range(len(qsm_files))], 'seg_files': gre_seg_clean_files },
    'T1' : { 'in_files' : [f"{t1_resampled_files[i]}" for i in range(len(qsm_files))], 'seg_files': gre_seg_clean_files },
    'T2s' : { 'in_files' : [f"{t2s_files[i]}" for i in range(len(qsm_files))], 'seg_files': gre_seg_clean_files },
    'R2s' : { 'in_files' : [f"{r2s_files[i]}" for i in range(len(qsm_files))], 'seg_files': gre_seg_clean_files },
    'GRE' : { 'in_files' : [f"{mag_files[i]}" for i in range(len(qsm_files))], 'seg_files': gre_seg_clean_files },
}

# %%
class AugmentMarkers(fastcore.transform.ItemTransform):

    inverted_ids = []

    def encodes(self, xy):
        # convert data
        x, y = xy
        x = x.numpy()
        y_np = np.array(y.numpy() == 1, dtype=int)[0,:,:,:]

        # determine connected regions
        labels, nlabels = scipy.ndimage.label(y_np, structure=scipy.ndimage.generate_binary_structure(3, 3))

        # invert some markers with 5% probability
        for i in range(nlabels):
            if random.random() <= 0.05:
                self.inverted_ids.append(i+1)
                x[0,labels == i+1] *= -1

        return (fastMONAI.vision_all.MedImage(x), y)

# %%
model_name = sys.argv[1]
fold_id = int(sys.argv[2])
k_folds = 24
random_state = 42
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S')

batch_size = 6
training_epochs = 700
lr = 0.003
ce_loss_weights = torch.Tensor([1, 1, 1])
training_augmentations = [
    fastMONAI.vision_all.PadOrCrop([80, 80, 80]),
    fastMONAI.vision_all.RandomFlip(axes=("LR",)),
    fastMONAI.vision_all.RandomFlip(axes=("AP",)),
    fastMONAI.vision_all.RandomAffine(degrees=(90, 90, 90)),
    fastMONAI.vision_all.ZNormalization()
]

if 'QSM' in model_name: training_augmentations.append(AugmentMarkers())

# %%
# split training/testing
df = pd.DataFrame(model_data[model_name])

# determine resampling suggestion
med_dataset = fastMONAI.vision_all.MedDataset(
    img_list=df.seg_files.tolist(),
    dtype=fastMONAI.vision_all.MedMask
)
suggested_voxelsize, requires_resampling = med_dataset.suggestion()
largest_imagesize = med_dataset.get_largest_img_size(resample=suggested_voxelsize)

# train over k validation folds
kf = KFold(n_splits=k_folds, random_state=random_state, shuffle=True)
(train_index, valid_index) = list(kf.split(df))[fold_id]

# prepare dataloader
dls = fastMONAI.vision_all.DataLoaders.from_dblock(
    fastMONAI.vision_all.MedDataBlock(
        blocks=(fastMONAI.vision_all.ImageBlock(cls=fastMONAI.vision_all.MedImage), fastMONAI.vision_all.MedMaskBlock),
        splitter=fastMONAI.vision_all.IndexSplitter(valid_index),
        get_x=fastMONAI.vision_all.ColReader('in_files'),
        get_y=fastMONAI.vision_all.ColReader('seg_files'),
        item_tfms=training_augmentations,
        reorder=requires_resampling,
        resample=suggested_voxelsize
    ),
    df,
    bs=batch_size
)

# prepare model
learn = fastMONAI.vision_all.Learner(
    dls,
    model=UNet(
        spatial_dims=3,
        in_channels=len(model_data[model_name]['in_files'][0].split(';')),  # qsm
        out_channels=3, # background, marker, calcification
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

start_time = time.time()
learn.fit_flat_cos(
    training_epochs,
    lr,
    cbs=[
        fastMONAI.vision_all.SaveModelCallback(monitor='valid_loss', fname=f"{model_name}-{timestamp}-{fold_id}-best"),
        fastMONAI.vision_all.EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=200)
    ]
)
end_time = time.time()
duration_mins = (end_time - start_time) / 60
print(f"Finished training after {round(duration_mins, 2)} mins")
learn.save(f"{model_name}-{timestamp}-{fold_id}-final")

#learn.recorder.plot_loss()

# %%
#learn = learn.load('QSM-T1-NOWEIGHT-20230526-152731-best')
#learn.model.cuda()


