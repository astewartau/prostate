import torch
import numpy as np
from matplotlib import pyplot as plt

def get_slice(volume, dim=0, slice_idx=None):
    if slice_idx is None:
        slice_idx = volume.shape[dim] // 2
    idx_tuple = tuple(slice_idx if i == dim else slice(None) for i in range(len(volume.shape)))
    return volume[idx_tuple]

def show_histogram(data, mask=None, title=None, dim=0, n_bins=100, n_ticks=10, vrange_hist=None, vrange_ylim=None, vrange_imshow=None, cmap='gray', **args):
    # fix data
    if not isinstance(data, torch.Tensor): data = torch.tensor(data)

    # get slice
    slice = get_slice(data, dim=dim) if data.dim() == 3 else data
    if mask is not None: mask_slice = get_slice(mask) if mask.dim() == 3 else mask
    
    # superfigure
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_figwidth(15)
    if title: fig.suptitle(title)#fontsize=29

    # data subplot
    if vrange_imshow is None: vrange_imshow = (float(data.min()), float(data.max()))
    ax1.set_title = "Data"
    im = ax1.imshow(slice, vmin=vrange_imshow[0], vmax=vrange_imshow[1], cmap=cmap, **args)
    if mask is not None: ax1.imshow(mask_slice, cmap='jet', interpolation='none', alpha=1.0*(mask_slice>0))
    ax1.set_axis_off()
    plt.colorbar(im, ax=ax1)#, orientation='horizontal')#, pad=0.2)

    # histogram subplot
    if vrange_hist is None: vrange_hist = (float(data.min()), float(data.max()))
    hist = torch.histc(data, bins=n_bins, min=vrange_hist[0], max=vrange_hist[1])
    ax2.set_title = "Histogram"
    ax2.bar(range(len(hist)), hist, align='center', color='skyblue')
    if vrange_ylim: ax2.set_ylim(vrange_ylim[0], vrange_ylim[1])
    if mask is not None:
        hist2 = torch.histc(data*mask, bins=n_bins, min=vrange_hist[0], max=vrange_hist[1])
        hist2 *= (max(hist) / max(hist2)) * 0.1
        ax2.bar(range(len(hist2)), hist2, align='center', color='red')
    
    ticks = np.array(np.round(np.linspace(start=0, stop=len(hist), num=n_ticks, endpoint=False)))
    labels = [round(float(vrange_hist[0] + i*((vrange_hist[1] - vrange_hist[0]) / n_ticks)),3) for i in range(n_ticks)]
    ax2.set_xticks(ticks=ticks, labels=labels)

    def get_closest_tick(val): return np.argmin(np.abs(np.array(actual_ticks)-np.array(val)))

    actual_ticks = np.linspace(start=vrange_hist[0], stop=vrange_hist[1], num=n_bins)#.tolist()
    l1 = ax2.axvline(x=get_closest_tick(vrange_imshow[0]), color='b')
    l2 = ax2.axvline(x=get_closest_tick(vrange_imshow[1]), color='b')
    l3 = ax2.axvline(x=get_closest_tick(data.mean()), color='r')
    l4 = ax2.axvline(x=get_closest_tick(data.mean() + data.std()), color='darkred')
    l5 = ax2.axvline(x=get_closest_tick(data.mean() - data.std()), color='darkred')
    trans = ax2.get_xaxis_transform()

    plt.text(get_closest_tick(vrange_imshow[0]), .85, f"{round(vrange_imshow[0],3)}", transform=trans, horizontalalignment='left' if vrange_imshow[0] <= vrange_hist[0] else 'right')
    plt.text(get_closest_tick(vrange_imshow[1]), .85, f"{round(vrange_imshow[1],3)}", transform=trans, horizontalalignment='right' if vrange_imshow[1] >= vrange_hist[1] else 'left')
    plt.text(get_closest_tick(data.mean()), .9, f"{round(float(data.mean()),3)}±{round(float(data.std()),3)}", transform=trans, horizontalalignment='left')
    
    plt.show()
    plt.close()

def show_image(data, title=None, cmap='gray', **args):
    plt.figure()
    plt.imshow(data, cmap=cmap, **args)
    if title: plt.title(title)
    plt.axis('off')
    plt.show()
    plt.close()

    