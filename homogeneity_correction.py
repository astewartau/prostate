#!/usr/bin/env python3
import os
import sys
import nibabel as nib
import SimpleITK as sitk

def apply_homogeneity_correction(image_data):
    # Convert NumPy array to a SimpleITK image.
    sitk_image = sitk.GetImageFromArray(image_data)
    # Create a mask using Otsu thresholding.
    mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    # Perform N4 bias field correction.
    corrected_image = sitk.N4BiasFieldCorrection(sitk_image, mask)
    # Convert the corrected image back to a NumPy array.
    corrected_data = sitk.GetArrayFromImage(corrected_image)
    return corrected_data

def append_suffix_to_filename(filename, suffix):
    if filename.endswith('.nii.gz'):
        base = filename[:-7]
        return base + suffix + '.nii.gz'
    else:
        base, ext = os.path.splitext(filename)
        return base + suffix + ext

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: homogeneity_correction.py <T1w_image>")
        sys.exit(1)
    
    t1_file = sys.argv[1]
    print(f"Processing {t1_file}")
    
    nii = nib.load(t1_file)
    data = nii.get_fdata()
    corrected_data = apply_homogeneity_correction(data)
    
    new_file = append_suffix_to_filename(t1_file, '_homogeneity-corrected')
    corrected_img = nib.Nifti1Image(corrected_data, nii.affine, nii.header)
    nib.save(corrected_img, new_file)
    print(f"Saved corrected image to {new_file}")
