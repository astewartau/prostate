#!/usr/bin/env python3

import glob
import nibabel as nib
import numpy as np
import os
import seaborn as sns
import pandas as pd

from enum import Enum
from matplotlib import pyplot as plt
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_erosion, gaussian_filter, binary_opening
from scipy.stats import ttest_ind


class SegType(Enum):
    NO_LABEL = 0
    PROSTATE = 1
    GOLD_SEED = 2
    CALCIFICATION = 3


def clean_masks():
    # identify relevant QSM and segmentation files
    script_path = os.path.dirname(os.path.abspath(__file__))
    qsm_files = glob.glob(f"{script_path}/bids/sub-*/ses-*/extra_data/*qsm.*")
    seg_files = glob.glob(f"{script_path}/bids/sub-*/ses-*/extra_data/*segmentation.*")
    assert(len(qsm_files) == len(seg_files))
    print(f"Found {len(qsm_files)} QSM and segmentation files!")
    
    for i in range(len(qsm_files)):
        # load files
        print(f"Cleaning up mask {i+1} of {len(qsm_files)}...")
        qsm_nii = nib.load(qsm_files[i])
        seg_nii = nib.load(seg_files[i])

        # get image data
        qsm = qsm_nii.get_fdata()
        seg = np.array(seg_nii.get_fdata(), dtype=np.uint8)

        # separate prostate tissue values
        prostate_values = qsm[seg == SegType.PROSTATE.value]

        # identify susceptibility values less than two standard deviations below the mean
        inliers = qsm + 3*np.std(prostate_values) > np.mean(qsm)

        # clean up segmentations such that 'inliers' are excluded
        seg[np.logical_and(seg == SegType.GOLD_SEED.value, inliers)] = 0
        seg[np.logical_and(seg == SegType.CALCIFICATION.value, inliers)] = 0

        # save result using original file extension
        extension = ".".join(seg_files[i].split('.')[1:])
        nib.save(nib.Nifti1Image(seg, header=seg_nii.header, affine=seg_nii.affine), f"{seg_files[i].split('.')[0]}_clean.{extension}")

def create_qsm_overlays():
    # identify relevant QSM and CLEAN segmentation files
    script_path = os.path.dirname(os.path.abspath(__file__))
    qsm_files = glob.glob(f"{script_path}/bids/sub-*/ses-*/extra_data/*qsm.*")
    seg_files = glob.glob(f"{script_path}/bids/sub-*/ses-*/extra_data/*segmentation.*")
    assert(len(qsm_files) == len(seg_files))
    print(f"Found {len(qsm_files)} QSM and segmentation files!")
    
    # extract values from QSM
    for i in range(len(qsm_files)):
        # load files
        print(f"Reading qsm={os.path.split(qsm_files[i])[1]}; seg={os.path.split(seg_files[i])[1]}...")
        qsm_nii = nib.load(qsm_files[i])
        seg_nii = nib.load(seg_files[i])

        # get image data
        qsm = qsm_nii.get_fdata()
        seg = np.array(seg_nii.get_fdata(), dtype=np.uint8)

        # only keep gold seed / calcification regions
        qsm_overlay = qsm * np.logical_or(seg == SegType.GOLD_SEED.value, seg == SegType.CALCIFICATION.value)

        # save result
        extension = ".".join(seg_files[i].split('.')[1:])
        nib.save(nib.Nifti1Image(qsm_overlay, header=qsm_nii.header, affine=qsm_nii.affine), f"{qsm_files[i].split('.')[0]}_overlay.{extension}")


def create_auto_qsm_overlays():
    # identify relevant QSM and CLEAN segmentation files
    script_path = os.path.dirname(os.path.abspath(__file__))
    qsm_files = glob.glob(f"{script_path}/bids/sub-*/ses-*/extra_data/*qsm.*")
    seg_files = glob.glob(f"{script_path}/bids/sub-*/ses-*/extra_data/*segmentation.*")
    assert(len(qsm_files) == len(seg_files))
    print(f"Found {len(qsm_files)} QSM and segmentation files!")
    
    # extract values from QSM
    for i in range(len(qsm_files)):
        # load files
        print(f"Reading qsm={os.path.split(qsm_files[i])[1]}; seg={os.path.split(seg_files[i])[1]}...")
        qsm_nii = nib.load(qsm_files[i])
        seg_nii = nib.load(seg_files[i])

        # get image data
        qsm = qsm_nii.get_fdata()
        seg = np.array(seg_nii.get_fdata(), dtype=np.uint8)

        # separate prostate tissue values
        prostate_values = qsm[seg == SegType.PROSTATE.value]

        # identify susceptibility values less than two standard deviations below the mean
        outlier_mask = qsm + 2*np.std(prostate_values) < np.mean(qsm)
        outlier_mask_open = binary_opening(outlier_mask)
        outlier_mask_open_dilate = binary_dilation(binary_dilation(binary_dilation(outlier_mask_open)))
        outlier_mask_final = np.logical_or(outlier_mask_open, np.logical_and(outlier_mask, outlier_mask_open_dilate))

        # clean up segmentations such that 'inliers' are excluded
        qsm_overlay = qsm * outlier_mask_final

        # save result
        extension = ".".join(seg_files[i].split('.')[1:])
        nib.save(nib.Nifti1Image(qsm_overlay, header=qsm_nii.header, affine=qsm_nii.affine), f"{qsm_files[i].split('.')[0]}_auto-overlay.{extension}")

def create_qsm_t2star_boxplots():
    # identify relevant QSM and CLEAN segmentation files
    script_path = os.path.dirname(os.path.abspath(__file__))
    qsm_files = glob.glob(f"{script_path}/bids/sub-*/ses-*/extra_data/*qsm.*")
    seg_files = glob.glob(f"{script_path}/bids/sub-*/ses-*/extra_data/*segmentation*clean.*")
    t2s_files = glob.glob(f"{script_path}/bids/sub-*/ses-*/extra_data/*t2star*")
    assert(len(qsm_files) == len(seg_files) == len(t2s_files))
    print(f"Found {len(qsm_files)} QSM, segmentation, and t2star files!")
    
    # initialise arrays for extracted values
    prostate_values_qsm = np.array([])
    goldseed_values_qsm = np.array([])
    calcification_values_qsm = np.array([])
    prostate_values_t2s = np.array([])
    goldseed_values_t2s = np.array([])
    calcification_values_t2s = np.array([])

    # extract values from QSM
    for i in range(len(qsm_files)):
        # load files
        print(f"Reading qsm={os.path.split(qsm_files[i])[1]}; t2s={os.path.split(t2s_files[i])[1]}; seg={os.path.split(seg_files[i])[1]}...")
        qsm_nii = nib.load(qsm_files[i])
        t2s_nii = nib.load(t2s_files[i])
        seg_nii = nib.load(seg_files[i])

        # get image data
        qsm = qsm_nii.get_fdata()
        t2s = t2s_nii.get_fdata()
        seg = np.array(seg_nii.get_fdata(), dtype=np.uint8)

        # extract new QSM values
        new_prostate_values_qsm = qsm[seg == SegType.PROSTATE.value].flatten()
        new_goldseed_values_qsm = qsm[seg == SegType.GOLD_SEED.value].flatten()
        new_calcification_values_qsm = qsm[seg == SegType.CALCIFICATION.value].flatten()

        # reference QSM values to the prostate tissue
        new_prostate_values_qsm      -= np.mean(new_prostate_values_qsm)
        new_goldseed_values_qsm      -= np.mean(new_prostate_values_qsm)
        new_calcification_values_qsm -= np.mean(new_prostate_values_qsm)

        # extract new t2s values
        new_prostate_values_t2s = t2s[np.logical_and(seg == SegType.PROSTATE.value, np.logical_and(t2s > 0, t2s < 50, np.invert(np.isnan(t2s))))].flatten()
        new_goldseed_values_t2s = t2s[np.logical_and(seg == SegType.GOLD_SEED.value, np.logical_and(t2s > 0, t2s < 50, np.invert(np.isnan(t2s))))].flatten()
        new_calcification_values_t2s = t2s[np.logical_and(seg == SegType.CALCIFICATION.value, np.logical_and(t2s > 0, t2s < 50, np.invert(np.isnan(t2s))))].flatten()

        # store values
        prostate_values_qsm = np.concatenate((prostate_values_qsm, new_prostate_values_qsm))
        goldseed_values_qsm = np.concatenate((goldseed_values_qsm, new_goldseed_values_qsm))
        calcification_values_qsm = np.concatenate((calcification_values_qsm, new_calcification_values_qsm))
        prostate_values_t2s = np.concatenate((prostate_values_t2s, new_prostate_values_t2s))
        goldseed_values_t2s = np.concatenate((goldseed_values_t2s, new_goldseed_values_t2s))
        calcification_values_t2s = np.concatenate((calcification_values_t2s, new_calcification_values_t2s))

    print("T-TEST - QSM CALC vs. GOLD-SEED", ttest_ind(calcification_values_qsm, goldseed_values_qsm))
    print("T-TEST - QSM PROSTATE vs. GOLD-SEED", ttest_ind(prostate_values_qsm, goldseed_values_qsm))
    print("T-TEST - T2s CALC vs. GOLD-SEED", ttest_ind(calcification_values_t2s, goldseed_values_t2s))
    print("T-TEST - T2s PROSTATE vs. GOLD-SEED", ttest_ind(prostate_values_t2s, goldseed_values_t2s))

    # prepare dataframe
    prostate_values_qsm = np.concatenate((
        np.full((len(prostate_values_qsm), 1), 'Prostate'),
        prostate_values_qsm.reshape(-1,1)
    ), axis=1)

    goldseed_values_qsm = np.concatenate((
        np.full((len(goldseed_values_qsm), 1), 'Gold seed'),
        goldseed_values_qsm.reshape(-1,1)
    ), axis=1)

    calcification_values_qsm = np.concatenate((
        np.full((len(calcification_values_qsm), 1), 'Calcification'),
        calcification_values_qsm.reshape(-1,1)
    ), axis=1)

    prostate_values_t2s = np.concatenate((
        np.full((len(prostate_values_t2s), 1), 'Prostate'),
        prostate_values_t2s.reshape(-1,1)
    ), axis=1)

    goldseed_values_t2s = np.concatenate((
        np.full((len(goldseed_values_t2s), 1), 'Gold seed'),
        goldseed_values_t2s.reshape(-1,1)
    ), axis=1)

    calcification_values_t2s = np.concatenate((
        np.full((len(calcification_values_t2s), 1), 'Calcification'),
        calcification_values_t2s.reshape(-1,1)
    ), axis=1)

    # bring all QSM values together into a dataframe
    all_qsm_values = np.concatenate((prostate_values_qsm, goldseed_values_qsm, calcification_values_qsm))
    df_qsm = pd.DataFrame(data=all_qsm_values, columns=['Region', 'Susceptibility (ppm)'])
    df_qsm['Susceptibility (ppm)'] = pd.to_numeric(df_qsm['Susceptibility (ppm)'])

    # create QSM figure
    ax = sns.violinplot(x='Region', y='Susceptibility (ppm)', data=df_qsm, color='palegreen')
    ax.set_title(f"Susceptibility values across manually segmented prostate ROIs (n={len(qsm_files)})\n(TGV-QSM; two-pass)")
    ax.set(ylim=(-3, +1.25))
    plt.savefig('boxplot-qsm.png', bbox_inches='tight', dpi=200)
    plt.close()

    # bring all t2s values together into a dataframe
    all_t2s_values = np.concatenate((prostate_values_t2s, goldseed_values_t2s, calcification_values_t2s))
    df_t2s = pd.DataFrame(data=all_t2s_values, columns=['Region', 'T2* time (ms)'])
    df_t2s['T2* time (ms)'] = pd.to_numeric(df_t2s['T2* time (ms)'])

    # create QSM figure
    ax = sns.violinplot(x='Region', y='T2* time (ms)', data=df_t2s, color='palegreen')
    ax.set_title(f"T2* values across manually segmented prostate ROIs (n={len(t2s_files)})")
    plt.savefig('boxplot-t2s.png', bbox_inches='tight', dpi=200)
    plt.close()

if __name__ == "__main__":
    clean_masks()
    #create_qsm_t2star_boxplots()
    #create_qsm_overlays()
    #create_auto_qsm_overlays()

