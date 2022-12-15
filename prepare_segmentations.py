import nibabel as nib
import numpy as np
import glob

seg_files = glob.glob("data/bids/sub-*/ses-*/extra_data/*segmentation_clean.nii*")

for seg_file in seg_files:
    nii = nib.load(seg_file)
    seg = nii.get_fdata()
    seg = np.array(seg == 2, dtype=seg.dtype)
    nib.save(nib.Nifti1Image(seg, header=nii.header, affine=nii.affine), f"{seg_file.split('.')[0]}_seeds.nii")

for seg_file in seg_files:
    nii = nib.load(seg_file)
    seg = nii.get_fdata()
    seg[seg == 1] = 0 # prostate
    seg[seg == 2] = 1 # seed
    seg[seg == 3] = 2 # calc
    nib.save(nib.Nifti1Image(seg, header=nii.header, affine=nii.affine), f"{seg_file.split('.')[0]}_seeds-calc.nii")

