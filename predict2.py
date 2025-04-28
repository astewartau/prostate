#!/usr/bin/env python3
import os
import argparse
import torch
import nibabel as nib
import torchio as tio
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, channels=(16, 32, 64, 128, 256)):
        super(UNet3D, self).__init__()
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        prev_channels = in_channels
        for ch in channels:
            self.encoders.append(self.conv_block(prev_channels, ch))
            prev_channels = ch
        self.bottleneck = self.conv_block(prev_channels, prev_channels * 2)
        rev_channels = list(reversed(channels))
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        cur_channels = prev_channels * 2
        for ch in rev_channels:
            self.upconvs.append(nn.ConvTranspose3d(cur_channels, ch, kernel_size=2, stride=2))
            self.decoders.append(self.conv_block(ch * 2, ch))
            cur_channels = ch
        self.final_conv = nn.Conv3d(cur_channels, 3, kernel_size=1)
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        enc_feats = []
        for encoder in self.encoders:
            x = encoder(x)
            enc_feats.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upconv, decoder, enc in zip(self.upconvs, self.decoders, reversed(enc_feats)):
            x = upconv(x)
            if x.shape != enc.shape:
                diffZ = enc.size()[2] - x.size()[2]
                diffY = enc.size()[3] - x.size()[3]
                diffX = enc.size()[4] - x.size()[4]
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2,
                              diffZ // 2, diffZ - diffZ // 2])
            x = torch.cat([enc, x], dim=1)
            x = decoder(x)
        return self.final_conv(x)


# your MergeInputChannels from training
class MergeInputChannels(tio.Transform):
    def __init__(self, infile_cols):
        super().__init__()
        self.infile_cols = infile_cols
    def apply_transform(self, subject):
        tensors = []
        for col in self.infile_cols:
            img = subject.get(col)
            if img is None: continue
            data = img.data
            if data.ndim == 3:
                data = data.unsqueeze(0)
            tensors.append(data)
        merged = torch.cat(tensors, dim=0)
        subject['image'] = tio.ScalarImage(
            tensor=merged, affine=subject[self.infile_cols[0]].affine
        )
        return subject

# compute how we cropped/padded
def get_crop_pad_params(orig_shape, proc_shape, target_shape=(100,100,64)):
    crop_params, pad_params = [None]*3, [None]*3
    for ax in range(3):
        od, pd = orig_shape[ax], proc_shape[ax]
        td = target_shape[ax]
        # center‐crop if larger
        if od >= td:
            start = (od - td)//2
            crop_params[ax] = (start, start+td)
        else:
            crop_params[ax] = None
        # pad if smaller
        if od < td:
            diff = td - od
            left = diff//2
            pad_params[ax] = (left, diff-left)
        else:
            pad_params[ax] = None
    return crop_params, pad_params

# undo crop+pad
def invert_crop_pad(arr, orig_shape, crop_params, pad_params):
    # first undo padding
    sl = []
    for ax in range(3):
        if pad_params[ax]:
            l,_ = pad_params[ax]
            sl.append(slice(l, l+orig_shape[ax]))
        else:
            sl.append(slice(None))
    truncated = arr[sl[0], sl[1], sl[2]]
    # then place into zeros of orig shape at crop location
    out = np.zeros(orig_shape, dtype=truncated.dtype)
    for ax in range(3):
        if crop_params[ax]:
            s,e = crop_params[ax]
            sl_out = [slice(None)]*3
            sl_out[ax] = slice(s,e)
        else:
            sl_out = [slice(None)]*3
        # combine all axes
    # better to build full sl_out list once:
    sl_out = []
    for ax in range(3):
        if crop_params[ax]:
            s,e = crop_params[ax]
            sl_out.append(slice(s,e))
        else:
            sl_out.append(slice(None))
    out[sl_out[0], sl_out[1], sl_out[2]] = truncated
    return out

def parse_args():
    p = argparse.ArgumentParser(description='3D-UNet inference with TorchIO preprocessing')
    p.add_argument('-i','--input',     required=True, help='Input NIfTI file')
    p.add_argument('-m','--model',     required=True, help='Trained .pth file')
    p.add_argument('-o','--output_dir',required=True, help='Where to save outputs')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # --- 1) build the same TorchIO pipeline ---
    infile_cols = ['t1_corrected']   # or your list
    transforms = tio.Compose([
        tio.CropOrPad((100,100,64)),
        tio.ZNormalization(),
        MergeInputChannels(infile_cols),
    ])

    # --- 2) load model ---
    model = UNet3D(in_channels=len(infile_cols), out_channels=3).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state if 'model_state_dict' not in state else state['model_state_dict'])
    model.eval()

    # --- 3) load input & record original shape/header ---
    img = nib.load(args.input)
    orig_np = img.get_fdata().astype(np.float32)
    orig_shape = orig_np.shape
    hdr = img.header

    # --- 4) wrap in a Subject and preprocess ---
    subj = tio.Subject(**{infile_cols[0]: tio.ScalarImage(args.input)})
    subj_t = transforms(subj)
    proc_np = subj_t['image'].data.squeeze(0).numpy()  # shape (100,100,64)

    # figure out crop/pad params
    crop_params, pad_params = get_crop_pad_params(orig_shape, proc_np.shape)

    # --- 5) inference ---
    tensor = torch.from_numpy(proc_np).unsqueeze(0).to(device)  # [1,D,H,W]
    with torch.no_grad():
        out = model(tensor.unsqueeze(0))[0]       # [3,100,100,64]
        probs = F.softmax(out, dim=0).cpu().numpy()

    # --- 6) invert & save ---
    for i in range(probs.shape[0]):
        inv = invert_crop_pad(probs[i], orig_shape, crop_params, pad_params)
        nib.save(nib.Nifti1Image(inv, img.affine, hdr),
                 os.path.join(args.output_dir, f"prob_class{i}.nii.gz"))
    seg = np.argmax(probs, axis=0).astype(np.uint8)
    seg_inv = invert_crop_pad(seg, orig_shape, crop_params, pad_params)
    nib.save(nib.Nifti1Image(seg_inv, img.affine, hdr),
             os.path.join(args.output_dir, "segmentation.nii.gz"))

    print("Done.", flush=True)

if __name__ == '__main__':
    main()