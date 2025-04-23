import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib


def parse_args():
    parser = argparse.ArgumentParser(description='Run 3D UNet segmentation on a NIfTI volume')
    parser.add_argument('-i', '--input', required=True,
                        help='Input NIfTI file path')
    parser.add_argument('-m', '--model', required=True,
                        help='Trained model .pth file')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Directory to save outputs')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')
    return parser.parse_args()


class UNet3D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, channels=(16,32,64,128,256)):
        super().__init__()
        self.pool = torch.nn.MaxPool3d(2)
        prev = in_channels
        self.encoders = torch.nn.ModuleList()
        for ch in channels:
            self.encoders.append(self.conv_block(prev, ch))
            prev = ch
        self.bottleneck = self.conv_block(prev, prev*2)
        rev = list(reversed(channels))
        self.upconvs = torch.nn.ModuleList()
        self.decoders = torch.nn.ModuleList()
        cur = prev*2
        for ch in rev:
            self.upconvs.append(torch.nn.ConvTranspose3d(cur, ch, 2, 2))
            self.decoders.append(self.conv_block(ch*2, ch))
            cur = ch
        # match training state dict keys
        self.final_conv = torch.nn.Conv3d(cur, out_channels, 1)

    def conv_block(self, in_ch, out_ch):
        return torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm3d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm3d(out_ch),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feats = []
        for enc in self.encoders:
            x = enc(x)
            feats.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, enc_feat in zip(self.upconvs, self.decoders, reversed(feats)):
            x = up(x)
            if x.shape != enc_feat.shape:
                diff = [e - x for e, x in zip(enc_feat.shape[2:], x.shape[2:])]
                pads = []
                for d in reversed(diff):
                    pads += [d//2, d - d//2]
                x = F.pad(x, pads)
            x = torch.cat([enc_feat, x], dim=1)
            x = dec(x)
        return self.final_conv(x)


def z_normalize(vol):
    mask = vol != 0
    vals = vol[mask]
    if vals.size == 0:
        return vol
    m, s = vals.mean(), vals.std()
    if s == 0:
        return vol - m
    return (vol - m) / s


def center_crop_or_pad(vol, size):
    orig = vol.shape
    pads = []
    for o, s in zip(orig, size):
        diff = max(0, s - o)
        pads.extend([diff//2, diff - diff//2])
    pad_cfg = [(pads[c*2], pads[c*2+1]) for c in range(3)]
    vol_p = np.pad(vol, pad_cfg, mode='constant')
    start = [(p[0] + (vol_p.shape[i] - size[i] - p[0] - p[1])//2) for i, p in enumerate(pad_cfg)]
    end = [st + size[i] for i, st in enumerate(start)]
    slices = tuple(slice(st, ed) for st, ed in zip(start, end))
    return vol_p[slices], orig, pad_cfg, start


def invert_crop_and_pad(pred, orig_shape, pad_cfg, crop_start):
    padded_shape = [orig_shape[i] + pad_cfg[i][0] + pad_cfg[i][1] for i in range(3)]
    padded = np.zeros(padded_shape, dtype=pred.dtype)
    cs = crop_start
    ce = [cs[i] + pred.shape[i] for i in range(3)]
    padded[cs[0]:ce[0], cs[1]:ce[1], cs[2]:ce[2]] = pred
    slices = tuple(slice(pad_cfg[i][0], pad_cfg[i][0] + orig_shape[i]) for i in range(3))
    return padded[slices]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    model = UNet3D(in_channels=1, out_channels=3)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    img = nib.load(args.input)
    vol = img.get_fdata().astype(np.float32)
    vol_n = z_normalize(vol)

    crop_size = (100, 100, 64)
    vol_crop, orig_shape, pad_cfg, crop_start = center_crop_or_pad(vol_n, crop_size)

    x = torch.from_numpy(vol_crop).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)[0].cpu().numpy()
    seg_crop = np.argmax(probs, axis=0)

    for cls in range(probs.shape[0]):
        p = probs[cls]
        p_full = invert_crop_and_pad(p, orig_shape, pad_cfg, crop_start)
        out_img = nib.Nifti1Image(p_full.astype(np.float32), img.affine, img.header)
        nib.save(out_img, os.path.join(args.output_dir, f'prob_class{cls}.nii.gz'))

    seg_full = invert_crop_and_pad(seg_crop, orig_shape, pad_cfg, crop_start)
    seg_img = nib.Nifti1Image(seg_full.astype(np.uint8), img.affine, img.header)
    nib.save(seg_img, os.path.join(args.output_dir, 'segmentation.nii.gz'))


if __name__ == '__main__':
    main()
