from glob import glob
from scipy.ndimage import label, binary_dilation
from scipy import stats
from useful_functions import *
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import numpy as np
import pandas as pd
import os
import json
import glob
import enum

class SegType(enum.Enum):
    NO_LABEL = 0
    GOLD_SEED = 1
    CALCIFICATION = 2
    PROSTATE = 3

def get_image_paths():
    bids_dir = "bids"

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

    extra_files = sorted(sum((glob.glob(os.path.join(session_dir, "extra_data", "*.nii*")) for session_dir in session_dirs), []))

    qsm_files2 = sorted(sum((glob.glob(os.path.join(session_dir, "extra_data", "*real.nii*")) for session_dir in session_dirs), []))
    qsm_files = [x for x in sorted(glob.glob(os.path.join("out/qsm/*.nii"))) if 'sub-z0449294' not in x]
    t2s_files = [x for x in sorted(glob.glob(os.path.join("out/t2s/*.nii"))) if 'sub-z0449294' not in x]
    r2s_files = [x for x in sorted(glob.glob(os.path.join("out/r2s/*.nii"))) if 'sub-z0449294' not in x]
    swi_files = [x for x in sorted(glob.glob(os.path.join("out/swi/*swi.nii"))) if 'sub-z0449294' not in x]
    mag_files = [extra_file for extra_file in extra_files if os.path.split(extra_file)[1] == 'magnitude_combined.nii']
    fmap_files = [extra_file for extra_file in extra_files if os.path.split(extra_file)[1] == 'B0.nii']

    gre_seg_clean_files = sorted([extra_file for extra_file in extra_files if all(pattern in extra_file for pattern in ['segmentation_clean.', 'tgvqsmjl'])])
    t1_files = sorted([extra_file for extra_file in extra_files if 'T1w_resliced' in extra_file])

    # Missing CT for sub-z0449294 
    ct_files = sorted([extra_file for extra_file in extra_files if 'resliced' in extra_file and 'T1w' not in extra_file and 'segmentation' not in extra_file and 'downsampled' not in extra_file])
    ct_seg_files = sorted([ct_file.replace(".nii", "_segmentation.nii") for ct_file in ct_files if os.path.exists(ct_file.replace(".nii", "_segmentation.nii"))])
    gre_seg_files = ct_seg_files
    ct_seg_clean_files = sorted([ct_file.replace(".nii", "_clean.nii") for ct_file in ct_seg_files if os.path.exists(ct_file.replace(".nii", "_clean.nii"))])

    print(f"{len(fmap_files)} field maps found.")
    print(f"{len(ct_files)} CT images found.")
    print(f"{len(ct_seg_files)} raw CT segmentations found.")
    print(f"{len(ct_seg_clean_files)} clean CT segmentations found.")
    print(f"{len(gre_seg_files)} raw GRE segmentations found.")
    print(f"{len(gre_seg_clean_files)} clean GRE segmentations found.")
    print(f"{len(qsm_files)} QSM images found.")
    print(f"{len(qsm_files2)} QSM (2) images found.")
    print(f"{len(mag_files)} magnitude images found.")
    print(f"{len(t2s_files)} T2* maps found.")
    print(f"{len(r2s_files)} R2* maps found.")
    print(f"{len(swi_files)} SWI maps found.")
    print(f"{len(t1_files)} T1w files found.")

    assert(len(qsm_files) == len(gre_seg_files))
    assert(len(qsm_files) == len(gre_seg_clean_files))
    assert(len(qsm_files) == len(fmap_files))
    assert(len(qsm_files) == len(t2s_files))
    assert(len(qsm_files) == len(r2s_files))
    assert(len(qsm_files) == len(swi_files))
    assert(len(qsm_files) == len(mag_files))
    assert(len(qsm_files) == len(t1_files))
    assert(len(qsm_files) == len(ct_files))
    assert(len(ct_files) == len(ct_seg_clean_files))
    assert(len(ct_files) == len(ct_seg_files))

    return {
        "qsm_files": qsm_files,
        "t2s_files": t2s_files,
        "r2s_files": r2s_files,
        "mag_files": mag_files,
        "swi_files": swi_files,
        "t1_files": t1_files,
        "fmap_files": fmap_files,
        "ct_files": ct_files,
        "ct_seg_files": ct_seg_files,
        "ct_seg_clean_files": ct_seg_clean_files,
        "gre_seg_files": gre_seg_files,
        "gre_seg_clean_files": gre_seg_clean_files
    }

def region_analysis(input_images, input_segmentations, raw_segmentations, clip_data=None):

    all_records = []

    for i, seg_file in enumerate(input_segmentations):
        # Load the image and segmentation data
        input_nii = nib.load(input_images[i])
        seg_nii = nib.load(seg_file)
        raw_seg_nii = nib.load(raw_segmentations[i])
        input_data = input_nii.get_fdata()
        seg = np.array(seg_nii.get_fdata(), dtype=np.uint8)
        raw_seg = np.array(raw_seg_nii.get_fdata(), dtype=np.uint8)

        # clip image data
        input_data[np.isnan(input_data)] = 0
        if clip_data is not None:
            input_data[input_data == np.inf] = clip_data[1]
            input_data[input_data < clip_data[0]] = clip_data[0]
            input_data[input_data > clip_data[1]] = clip_data[1]
        else:
            input_data[input_data == np.inf] = max(input_data[input_data != np.inf])

        prostate_mask = (raw_seg == 1)
        prostate_values = input_data[prostate_mask]
        record = {
            'seg_value': SegType.PROSTATE.value,
            'label_id': 1,
            'size': prostate_values.size,
            'mean': prostate_values.mean(),
            'std': prostate_values.std(),
            'segfile': os.path.split(seg_file)[1],
            'cropped_source': prostate_values,
            'cropped_source_mask': None,
        }
        all_records.append(record)

        # Iterate over unique segmentation values
        for seg_value in np.unique(seg[seg != 0]):
            labeled_mask, num_labels = label(
                input=np.array(seg == seg_value, dtype=np.uint8),
                structure=np.ones((3, 3, 3))
            )

            # For each distinct region
            for label_id in range(1, num_labels + 1):
                mask = (labeled_mask == label_id)
                region_values = input_data[mask]

                # get regions
                half_cropsize = 20 // 2
                coords = np.argwhere(mask)
                centroid = np.mean(coords, axis=0).astype(int)
                x_start, x_end = max(0, centroid[0]-half_cropsize), min(input_data.shape[0], centroid[0]+half_cropsize)
                y_start, y_end = max(0, centroid[1]-half_cropsize), min(input_data.shape[1], centroid[1]+half_cropsize)
                z_start, z_end = max(0, centroid[2]-half_cropsize), min(input_data.shape[2], centroid[2]+half_cropsize)
                submask = np.array(mask[x_start:x_end, y_start:y_end, z_start:z_end], dtype=int)
                subvals = np.array(input_data[x_start:x_end, y_start:y_end, z_start:z_end], dtype=float)

                record = {
                    'seg_value': seg_value,
                    'label_id': label_id,
                    'size': region_values.size,
                    'mean': region_values.mean(),
                    'std': region_values.std(),
                    'segfile': os.path.split(seg_file)[1],
                    'cropped_source': subvals,
                    'cropped_source_mask': submask,
                }
                
                all_records.append(record)

    return pd.DataFrame(all_records)

def create_source_figure(analysis_results, v_range, figpath):
    fig, axes = plt.subplots(ncols=6, nrows=len(analysis_results['cropped_source']), figsize=(12, 50))

    for ax in axes.flat:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(len(analysis_results)):
        try:
            axes[i,0].set_ylabel(analysis_results.iloc[i]['segfile'][5:17], rotation=0, fontsize=12, labelpad=55)
            axes[i,0].imshow(analysis_results.iloc[i]['cropped_source'][analysis_results.iloc[i]['cropped_source'].shape[0]//2,:,:], cmap='gray', vmin=v_range[0], vmax=v_range[1], interpolation='nearest')
            axes[i,1].imshow(analysis_results.iloc[i]['cropped_source'][:,analysis_results.iloc[i]['cropped_source'].shape[1]//2,:], cmap='gray', vmin=v_range[0], vmax=v_range[1], interpolation='nearest')
            axes[i,2].imshow(analysis_results.iloc[i]['cropped_source'][:,:,analysis_results.iloc[i]['cropped_source'].shape[2]//2], cmap='gray', vmin=v_range[0], vmax=v_range[1], interpolation='nearest')

            axes[i,3].imshow(analysis_results.iloc[i]['cropped_source'][analysis_results.iloc[i]['cropped_source'].shape[0]//2,:,:], cmap='gray', vmin=v_range[0], vmax=v_range[1], interpolation='nearest')
            axes[i,4].imshow(analysis_results.iloc[i]['cropped_source'][:,analysis_results.iloc[i]['cropped_source'].shape[1]//2,:], cmap='gray', vmin=v_range[0], vmax=v_range[1], interpolation='nearest')
            axes[i,5].imshow(analysis_results.iloc[i]['cropped_source'][:,:,analysis_results.iloc[i]['cropped_source'].shape[2]//2], cmap='gray', vmin=v_range[0], vmax=v_range[1], interpolation='nearest')

            axes[i,3].imshow(analysis_results.iloc[i]['cropped_source_mask'][analysis_results.iloc[i]['cropped_source'].shape[0]//2,:,:], cmap='cool', alpha=np.array(analysis_results.iloc[i]['cropped_source_mask'][analysis_results.iloc[i]['cropped_source'].shape[0]//2,:,:] * 0.2, dtype=float), vmin=0, vmax=1)
            axes[i,4].imshow(analysis_results.iloc[i]['cropped_source_mask'][:,analysis_results.iloc[i]['cropped_source'].shape[1]//2,:], cmap='cool', alpha=np.array(analysis_results.iloc[i]['cropped_source_mask'][:,analysis_results.iloc[i]['cropped_source'].shape[1]//2,:] * 0.2, dtype=float), vmin=0, vmax=1)
            axes[i,5].imshow(analysis_results.iloc[i]['cropped_source_mask'][:,:,analysis_results.iloc[i]['cropped_source'].shape[2]//2], cmap='cool', alpha=np.array(analysis_results.iloc[i]['cropped_source_mask'][:,:,analysis_results.iloc[i]['cropped_source'].shape[2]//2] * 0.2, dtype=float), vmin=0, vmax=1)
        except:
            print("Failed", i, analysis_results.iloc[i]['segfile'])

    plt.savefig(figpath, bbox_inches='tight', dpi=400)
    plt.close()

def main():
    image_paths = get_image_paths()
    
    for (input_images, input_segmentations, raw_segmentations, clip_data, v_range, name) in [
        #(image_paths['t1_files'], image_paths['gre_seg_clean_files'], image_paths['gre_seg_files'], None, (0, 800), "T1-weighted"),
        (image_paths['fmap_files'], image_paths['gre_seg_clean_files'], image_paths['gre_seg_files'], None, (-0.1, +0.1), "Field map"),
        #(image_paths['qsm_files'], image_paths['gre_seg_clean_files'], image_paths['gre_seg_files'], None, (-1, +1), "QSM (ppm)"),
        #(image_paths['t2s_files'], image_paths['gre_seg_clean_files'], image_paths['gre_seg_files'], (0, 100), (0, 100), "T2* (s)"),
        #(image_paths['r2s_files'], image_paths['gre_seg_clean_files'], image_paths['gre_seg_files'], (0, 100), (0, 100), "R2*"),
        #(image_paths['mag_files'], image_paths['gre_seg_clean_files'], image_paths['gre_seg_files'], None, (0, 800), "Magnitude"),
        #(image_paths['swi_files'], image_paths['gre_seg_clean_files'], image_paths['gre_seg_files'], None, (0, 0.75), "SWI"),
        #(image_paths['ct_files'], image_paths['ct_seg_clean_files'], image_paths['gre_seg_files'], None, (0, 1000), "CT (HU)"),
    ]:
        print(f"== {name.upper()} ==")

        analysis_results = region_analysis(input_images, input_segmentations, raw_segmentations, clip_data)

        print(analysis_results)

        if name.upper() == 'QSM':

            # == GOLD MARKER SIZE ==
            sns.histplot(analysis_results[analysis_results['seg_value'] == SegType.GOLD_SEED.value], x='size', discrete=False)
            plt.xlabel("Size (voxels)")
            plt.ylabel("FMs (#)")
            max_y = int(plt.gca().get_ylim()[1])
            yticks = [i for i in range(0, max_y+1, 2) if i % 2 == 0]
            plt.yticks(yticks)
            plt.savefig(f"seed_size_histogram.png", bbox_inches='tight', dpi=300)
            plt.close()

            # == CALC SIZE ==
            sns.histplot(analysis_results[analysis_results['seg_value'] == SegType.CALCIFICATION.value], x='size', discrete=False)
            plt.xlabel("Size (voxels)")
            plt.ylabel("Calcifications (#)")
            max_y = int(plt.gca().get_ylim()[1])
            yticks = [i for i in range(0, max_y+1, 2) if i % 2 == 0]
            plt.yticks(yticks)
            plt.savefig("calc_size_histogram.png", bbox_inches='tight', dpi=300)
            plt.close()

        # == SEED FIGURE == 
        create_source_figure(
            analysis_results=analysis_results[analysis_results['seg_value'] == SegType.GOLD_SEED.value],
            v_range=v_range,
            figpath=f"seeds_{name}.png"
        )
        
        # == CALC FIGURE ==
        create_source_figure(
            analysis_results=analysis_results[analysis_results['seg_value'] == SegType.CALCIFICATION.value],
            v_range=v_range,
            figpath=f"calcs_{name}.png"
        )

        # count the number of rows where the segtype is calcification
        calc_counts = analysis_results[analysis_results['seg_value'] == SegType.CALCIFICATION.value]['segfile'].value_counts()
        seed_counts = analysis_results[analysis_results['seg_value'] == SegType.GOLD_SEED.value]['segfile'].value_counts()

        print(seed_counts)

        seed_means = analysis_results[analysis_results['seg_value'] == SegType.GOLD_SEED.value]['mean']
        seed_size = analysis_results[analysis_results['seg_value'] == SegType.GOLD_SEED.value]['size']
        calc_means = analysis_results[
            (analysis_results['seg_value'] == SegType.CALCIFICATION.value) & 
            (analysis_results['size'] < seed_size.mean() + 2*seed_size.std()) & 
            (analysis_results['size'] > seed_size.mean() - 2*seed_size.std())
        ]['mean']
        prostate_means = analysis_results[analysis_results['seg_value'] == SegType.PROSTATE.value]['mean']

        # t-test
        t_stat, p_value = stats.ttest_ind(seed_means, calc_means)

        print('t-statistic:', t_stat)
        print('p-value:', p_value)

        print(f"calc == {calc_means.mean()} +/- {calc_means.std()}")
        print(f"seed == {seed_means.mean()} +/- {seed_means.std()}")

        # 1. Restructure the data:
        # Create a dataframe with two columns: 'type' for the group and 'mean_value' for the mean values.
        data = {
            'Region': ['FM'] * len(seed_means) + ['Calcification'] * len(calc_means) + ['Prostate'] * len(prostate_means),
            'Mean': pd.concat([seed_means, calc_means, prostate_means], axis=0),
            'dummy': ['A'] * len(seed_means) + ['A'] * len(calc_means) + ['A'] * len(prostate_means)
        }
        df_plot = pd.DataFrame(data)

        # 2. Plot the data:
        plt.figure(figsize=(6, 3))
        sns.violinplot(y='Region', x='Mean', data=df_plot, color='lightblue', split=True, density_norm='width')

        y1, y2 = 0, 1
        x = np.percentile(analysis_results['mean'], 97)
        width = (analysis_results['mean'].max() - analysis_results['mean'].min()) * 0.01
        plt.plot([x, x+width, x+width, x], [0, 0, 1, 1], lw=1.5, c='k')
        plt.text(x+width*2, 0.5, f"t={np.round(t_stat, 2)}, p={np.round(p_value, 2)}", ha='left', va='top')

        plt.xlabel(f"{name}")
        plt.savefig(f"values_{name}.png", bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    main()

