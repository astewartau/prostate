import random
import fastMONAI.vision_all
import fastcore.transform
import torch
import fastai
import numpy as np
import enum
import skimage.measure # label, regionprops
import scipy.ndimage # label
from matplotlib import pyplot as plt

class SegTypeRaw(enum.Enum):
    NO_LABEL = 0
    PROSTATE = 1
    GOLD_SEED = 2
    CALCIFICATION = 3

class SegTypeClean(enum.Enum):
    NO_LABEL = 0
    GOLD_SEED = 1
    CALCIFICATION = 2
    PROSTATE = 3

def get_centroids(mask):
    labeled_mask, num_labels = scipy.ndimage.label(
        input=mask,
        structure=np.ones((3, 3, 3))
    )

    centroids = []

    for region_label in range(1, num_labels+1):
        coords = np.argwhere(labeled_mask == region_label)

        centroid = np.mean(coords, axis=0)
        centroid = np.round(centroid).astype(int)
        centroids.append(centroid)

    return centroids

def get_cropped_regions(mask, vals, cropsize=10):
    
    labeled_mask, num_labels = scipy.ndimage.label(
        input=mask,
        structure=np.ones((3, 3, 3))
    )

    cropped_regions = []
    cropped_masks = []

    half_cropsize = cropsize // 2

    for region_label in range(1, num_labels+1):
        coords = np.argwhere(labeled_mask == region_label)
        centroid = np.mean(coords, axis=0).astype(int)

        x_start, x_end = centroid[0]-half_cropsize, centroid[0]+half_cropsize
        y_start, y_end = centroid[1]-half_cropsize, centroid[1]+half_cropsize
        z_start, z_end = centroid[2]-half_cropsize, centroid[2]+half_cropsize

        submask = np.array(mask[x_start:x_end, y_start:y_end, z_start:z_end], dtype=int)
        subvals = np.array(vals[x_start:x_end, y_start:y_end, z_start:z_end], dtype=float)

        cropped_regions.append(subvals)
        cropped_masks.append(submask)

    return cropped_regions, cropped_masks

def get_region_stats(seg, vals):
    counts = {}
    means = {}
    stds = {}

    for label_id in np.unique(seg):
        if label_id == 0: continue
        
        labels, num_labels = scipy.ndimage.label(
            seg * np.array((seg == label_id), int),
            structure=np.ones((3, 3, 3))
        )

        region_counts = []
        region_means = []
        region_stds = []
        for i in range(1, num_labels+1):
            mask = (labels == i)
            region_counts.append(np.sum(mask))
            region_means.append(np.mean(vals[mask]))
            region_stds.append(np.std(vals[mask]))

        counts[label_id] = region_counts
        means[label_id] = region_means
        stds[label_id] = region_stds

    return counts, means, stds

def get_center_slices(mask):
    labeled_mask = skimage.measure.label(mask)
    regions = skimage.measure.regionprops(labeled_mask)
    center_slices = [[round(coord) for coord in region.centroid] for region in regions]
    return center_slices

class SetVrange(fastMONAI.vision_all.DisplayedTransform):
    def __init__(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax

    def encodes(self, o:fastMONAI.vision_all.MedImage):
        o[o > self.vmax] = 0
        o[o < self.vmin] = 0
        return o
    
def show_images(x, y, figsize=None, fig_out=None):
    n_samples = x.shape[0]
    n_masks = y.shape[0]
    #assert(n_samples == n_masks)
    n_samples = max(1, n_masks)

    if y.shape[1] > 1:
        mask = torch.argmax(y, dim=1).unsqueeze(1).cpu().numpy()
    else:
        mask = y.cpu().numpy()
    mask = np.array(np.round(mask), dtype=int)
    data = x.cpu().numpy()

    max_sources = 1
    for i in range(n_samples):
        center_slices = get_center_slices(np.array(mask[i][0] == 1, dtype=int))
        n_sources = len(center_slices)
        max_sources = max(n_sources, max_sources)
    max_sources = min(7, max_sources)

    img_width = 2
    img_height = 2
    wspace = 0.05
    hspace = 0.05
    n_cols = max_sources
    n_rows = n_samples
    fig_width = img_width * n_cols + wspace * (n_cols - 1)
    fig_height = img_height * n_rows + hspace * (n_rows - 1)

    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(fig_width, fig_height), squeeze=False)
    
    for ax in axes.flat:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
    
    for i in range(n_rows):
        center_slices = get_center_slices(np.array(mask[i][0] == 1, dtype=int))
        n_sources = len(center_slices)

        for j in range(min(n_sources, n_cols)):
            axes[i,j].imshow(data[i][0][:,center_slices[j][1],:], cmap='gray', interpolation='nearest') 
            axes[i,j].imshow(mask[i][0][:,center_slices[j][1],:], cmap='Set1', vmin=1, vmax=9, alpha=np.array(mask[i][0][:,center_slices[j][1],:] != 0, dtype=int) * 0.6, interpolation='nearest')
    plt.tight_layout()
    if fig_out: plt.savefig(fig_out)
    plt.show()
    plt.close()
    
@fastMONAI.vision_all.typedispatch
def show_batch(x:fastMONAI.vision_all.MedImage, y:fastMONAI.vision_all.MedMask, samples, ctxs=None, max_n=6, nrows=None, ncols=2, figsize=None, **kwargs):
    show_images(x, y)

@fastMONAI.vision_all.typedispatch
def show_results(x:fastMONAI.vision_all.MedImage, y:fastMONAI.vision_all.MedMask, samples, outs, ctxs=None, max_n=6, nrows=None, ncols=2, figsize=None, fig_out='out.png', **kwargs):
    outs = torch.stack([outs[i][0] for i in range(len(outs))], dim=0)
    show_images(x, y, fig_out=f"{fig_out.split('.')[0]}_targ.png")
    show_images(x, outs, fig_out=f"{fig_out.split('.')[0]}_pred.png")
    

class MarkersIdentified(fastai.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.targ_marker_count = 0
        self.pred_marker_count = 0
        self.overlap_count = 0
    
    def reset(self):
        self.targ_marker_count = 0
        self.pred_marker_count = 0
        self.overlap_count = 0
    
    def accumulate(self, learn=None, pred=None, targ=None):
        if pred is None or targ is None:
            pred = learn.pred.argmax(dim=1).cpu().numpy()
            targ = learn.y.cpu().numpy()
        
        pred = np.array(np.round(pred) == 1, dtype=int)
        targ = np.array(np.round(targ) == 1, dtype=int)

        pred = scipy.ndimage.binary_dilation(pred)
        targ = scipy.ndimage.binary_dilation(targ)

        structure = np.ones((3, 3, 3)) == True

        for i in range(pred.shape[0]):
            if len(pred[i].shape) == 3:
                _, pred_nlabels = scipy.ndimage.label(pred[i], structure=structure)
                _, targ_nlabels = scipy.ndimage.label(targ[i][0], structure=structure)

                overlap = np.array(np.logical_and(pred[i] == targ[i][0], pred[i] == 1), dtype=int)
                _, n_overlaps = scipy.ndimage.label(overlap, structure=structure)
                
                self.pred_marker_count += pred_nlabels
                self.targ_marker_count += targ_nlabels
                self.overlap_count += n_overlaps
            else:
                for j in range(pred[i].shape[0]):
                    _, pred_nlabels = scipy.ndimage.label(pred[i][j], structure=structure)
                    _, targ_nlabels = scipy.ndimage.label(targ[i][j], structure=structure)
                
                    overlap = np.array(np.logical_and(pred[i][j] == targ[i][j], pred[i][j] == 1), dtype=int)
                    _, n_overlaps = scipy.ndimage.label(overlap, structure=structure)
                    
                    self.pred_marker_count += pred_nlabels
                    self.targ_marker_count += targ_nlabels
                    self.overlap_count += n_overlaps

    @property
    def value(self):
        return float(self.overlap_count) / max(1., float(self.targ_marker_count))

class SuperfluousMarkers(fastai.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.targ_marker_count = 0
        self.pred_marker_count = 0
        self.overlap_count = 0
    
    def reset(self):
        self.targ_marker_count = 0
        self.pred_marker_count = 0
        self.overlap_count = 0
    
    def accumulate(self, learn=None, pred=None, targ=None):
        if pred is None or targ is None:
            pred = learn.pred.argmax(dim=1).cpu().numpy()
            targ = learn.y.cpu().numpy()
        
        pred = np.array(np.round(pred), dtype=int)
        targ = np.array(np.round(targ), dtype=int)

        pred = scipy.ndimage.binary_dilation(pred)
        targ = scipy.ndimage.binary_dilation(targ)

        structure = np.ones((3, 3, 3)) == True

        for i in range(pred.shape[0]):
            if len(pred[i].shape) == 3:
                _, pred_nlabels = scipy.ndimage.label(pred[i], structure=structure)
                _, targ_nlabels = scipy.ndimage.label(targ[i][0], structure=structure)

                overlap = np.array(np.logical_and(pred[i] == targ[i][0], pred[i] == 1), dtype=int)
                _, n_overlaps = scipy.ndimage.label(overlap, structure=structure)
                
                self.pred_marker_count += pred_nlabels
                self.targ_marker_count += targ_nlabels
                self.overlap_count += n_overlaps
            else:
                for j in range(pred[i].shape[0]):
                    _, pred_nlabels = scipy.ndimage.label(pred[i][j], structure=structure)
                    _, targ_nlabels = scipy.ndimage.label(targ[i][j], structure=structure)
                
                    overlap = np.array(np.logical_and(pred[i][j] == targ[i][j], pred[i][j] == 1), dtype=int)
                    _, n_overlaps = scipy.ndimage.label(overlap, structure=structure)
                    
                    self.pred_marker_count += pred_nlabels
                    self.targ_marker_count += targ_nlabels
                    self.overlap_count += n_overlaps

    @property
    def value(self):
        return float(self.pred_marker_count - self.overlap_count) / max(1., float(self.pred_marker_count))

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

