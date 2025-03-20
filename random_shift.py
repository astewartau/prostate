#!/usr/bin/env python

import argparse
import nibabel as nib
import torch
import numpy as np
import torchio as tio

class RandomIntensityShift(tio.Transform):
    def __init__(self, gamma_range=(0.8, 1.2), p=0.5):
        super().__init__()
        self.gamma_range = gamma_range
        self.p = p

    def apply_transform(self, subject):
        if torch.rand(1).item() > self.p:
            return subject
        for image_name in subject.keys():
            if isinstance(subject[image_name], tio.ScalarImage):
                img_tensor = subject[image_name].data
                if img_tensor.ndim == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                gamma = np.random.uniform(*self.gamma_range)
                img_tensor = img_tensor.pow(gamma)
                subject[image_name].set_data(img_tensor)
        return subject

def apply_random_intensity_shift(input_path, output_path, gamma_range=(0.8, 1.2), p=1.0):
    """
    Applies a random intensity shift using gamma correction to a NIfTI image and saves the result.

    Parameters:
        input_path (str): Path to the input NIfTI file.
        output_path (str): Path to save the transformed NIfTI file.
        gamma_range (tuple): Range for the gamma value (default: (0.8, 1.2)).
        p (float): Probability of applying the transformation (default: 1.0, always applies).
    """
    print(f"Loading NIfTI file: {input_path}")
    nii = nib.load(input_path)
    image_data = nii.get_fdata()
    affine = nii.affine

    # Convert to TorchIO ScalarImage
    subject = tio.Subject(image=tio.ScalarImage(tensor=torch.tensor(image_data, dtype=torch.float32).unsqueeze(0), affine=affine))

    # Define the custom random intensity shift transformation
    transform = RandomIntensityShift(gamma_range=gamma_range, p=p)

    # Apply the transformation
    print("Applying random intensity shift...")
    transformed_subject = transform(subject)
    transformed_image = transformed_subject['image'].data.squeeze(0).numpy()

    # Save the transformed image
    nib.save(nib.Nifti1Image(transformed_image.astype(np.float32), affine), output_path)
    print(f"Saved transformed NIfTI to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a random intensity shift to a NIfTI image.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input NIfTI file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the transformed NIfTI file.")
    parser.add_argument("--gamma_min", type=float, default=0.8, help="Minimum gamma value for intensity shift (default: 0.8).")
    parser.add_argument("--gamma_max", type=float, default=1.2, help="Maximum gamma value for intensity shift (default: 1.2).")
    parser.add_argument("--probability", type=float, default=1.0, help="Probability of applying the transformation (default: 1.0).")

    args = parser.parse_args()

    apply_random_intensity_shift(args.input, args.output, gamma_range=(args.gamma_min, args.gamma_max), p=args.probability)
