import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib import rcParams
import math
import matplotlib.pylab as pl
import ot
import ot.plot
import matplotlib

config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "font.size": 20,
#     "mathtext.fontset":'stix',
}
rcParams.update(config)

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters  
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "black"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def draw_distribution(bins, weight_list, label_list, Wasserstein_distance, img_save_path):
    '''
    sl: the idx of selected layer
    lt: layer type (cw: conv weight, cb: conv bias, bw: batchnorm weight, bb: batchnorm bias)
    ''' 
    colors = ['red', 'orange', 'blue']
    fig, ax = plt.subplots(figsize = (7, 4))
    for i in range(len(weight_list)):
        ax.hist(weight_list[i], bins, label=label_list[i], density = True,  alpha = 0.6) 
        # ax.hist(weight_list[i], bins, label=label_list[i], density = True,  alpha = 0.5, color=colors[i]) 

    ax.set_xticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2])
    ax.set_xlim(left=-0.4, right=0.2)

    ax.set_xlabel('Non-zero Weight, bandwidth=0.01')
    ax.set_ylabel('Density')
    # ax.set_title('(a)')

    ####################################################################
    vegetables = ["#1", "#2", "#3", ]
    farmers = ["#1", "#2", "#3", ]
    cbarlabel="WD"

    harvest = np.array(Wasserstein_distance)

    left, bottom, width, height = 0.1, 0.38, 0.45, 0.45
    ax1 = fig.add_axes([left,bottom,width,height])

    im, cbar = heatmap(harvest, vegetables, farmers, ax=ax1,
                    cmap="Reds", cbarlabel="EMD")
    # ax1.set_title('(b)')
    texts = annotate_heatmap(im, valfmt="{x:.2f}",)
    # fig.tight_layout()

    # Tweak spacing to prevent clipping of ylabel
    ax.grid(ls='-.')
    ax.legend(fontsize=13, loc='upper right')
    plt.tight_layout()
    
    plt.savefig(img_save_path)


def draw_distribution_2(bins, weight_list, label_list, Wasserstein_distance, img_save_path):
    '''
    sl: the idx of selected layer
    lt: layer type (cw: conv weight, cb: conv bias, bw: batchnorm weight, bb: batchnorm bias)
    ''' 
    # colors = ['red', 'orange', 'blue']
    # for i in range(len(weight_list)):
    #     ax.hist(weight_list[i], bins, label=label_list[i], density = True,  alpha = 0.6) 
    #     # ax.hist(weight_list[i], bins, label=label_list[i], density = True,  alpha = 0.5, color=colors[i]) 

    # ax.set_xticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2])
    # ax.set_xlim(left=-0.4, right=0.2)

    # ax.set_xlabel('Non-zero Weight, bandwidth=0.01')
    # ax.set_ylabel('Density')
    # ax.set_title('(a)')

    ####################################################################
    fig, ax = plt.subplots(figsize = (7, 4))

    vegetables = ["#1", "#2", "#3", ]
    farmers = ["#1", "#2", "#3", ]

    Wasserstein_distance = [[0.,        1.544114,   0.40145464],
                            [1.544114,   0. ,        1.14265936],
                            [0.40145464, 1.14265936, 0.        ]]
    
    harvest = np.array(Wasserstein_distance)

    # left, bottom, width, height = 0.1, 0.38, 0.45, 0.45
    # ax1 = fig.add_axes([left,bottom,width,height])

    im, cbar = heatmap(harvest, vegetables, farmers, ax=ax,
                    cmap="Reds", cbarlabel="EMD")
    # ax1.set_title('(b)')
    texts = annotate_heatmap(im, valfmt="{x:.2f}",)
    # fig.tight_layout()

    # Tweak spacing to prevent clipping of ylabel
    # ax.grid(ls='-.')
    # ax.legend(fontsize=13, loc='upper right')
    plt.tight_layout()
    
    plt.savefig(img_save_path)
    

def extract_weight(model, sl, lt):
    '''
    extract the weight from the model's sl-th layer
    return the flatten version
    '''

    m_type = nn.Conv2d if 'c' in lt else nn.GroupNorm

    if sl == -1:  # return the weights of all layers
        flatten_weight = []
        for m in model.modules():
            if isinstance(m, m_type):
                if 'w' in lt:
                    flatten_weight += m.weight.view(-1).detach().cpu().numpy().tolist()  
                else:
                    flatten_weight += m.bias.view(-1).detach().cpu().numpy().tolist() 
    else:
        idx = 0
        for k, m in list(model.named_modules()):
            if isinstance(m, m_type):
                if idx == sl:
                    if 'w' in lt:
                        flatten_weight = m.weight.view(-1).detach().cpu().numpy().tolist()  
                    else:
                        flatten_weight = m.bias.view(-1).detach().cpu().numpy().tolist() 
                    break
                idx += 1
    
    
    flatten_weight = [i for i in flatten_weight if i != 0 and abs(i)<0.49]

    return np.array(flatten_weight)


def to_freq_list(distribution, num_bins, upper):
    freq_list = [0] * num_bins
    for i in distribution:
        freq_list[math.floor(100*(i+upper))]+=1
    freq_list = np.array(freq_list)
    return freq_list/freq_list.sum()


def calculate_KLD(P, Q):
    '''
    calculate the Kullback-Leibler Divergence(KLD) between distributions
    '''
    P_freq_list = to_freq_list(P)
    Q_freq_list = to_freq_list(Q)
    print(P_freq_list)
    print(Q_freq_list)


def calculate_WDS(distribution_list):
    '''
    calculate the Wasserstein distance(WD)/ Earth Mover Distance between distributions
    
    '''
    # print(max(P), min(P), max(Q), min(Q))
    upper = 0.50; lower = -0.50 # the range[lower, upper] is validated if it coule cover all the weight parameters in P and Q.
    num_bins = 100

    bins = np.arange(num_bins, dtype=np.float64)
    Cost_Matric = ot.dist(bins.reshape((num_bins, 1)), bins.reshape((num_bins, 1)), metric='euclidean')

    num_distribution = len(distribution_list)
    freq_list = []
    for i in range(num_distribution):
        freq_list.append(to_freq_list(distribution_list[i], num_bins, upper))
    # print(len(freq_list))

    WD_matirx = np.zeros((num_distribution, num_distribution))
    for i in range(num_distribution):
        for j in range(num_distribution):
            WD_matirx[i][j] = ot.emd2(freq_list[i], freq_list[j], Cost_Matric)

    return WD_matirx


# fig in fig
'''
import matplotlib.pyplot as plt
 
fig = plt.figure()
 
x = [1,2,3,4,5,6,7]
y = [1,3,4,2,5,8,6]
 
 
left, bottom, width, height = 0.1,0.1,0.8,0.8
ax1 = fig.add_axes([left,bottom,width,height])
ax1.plot(x,y,'r')
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title('title')
 
left, bottom, width, height = 0.2,0.6,0.25,0.25
ax2 = fig.add_axes([left,bottom,width,height])
ax2.plot(x,y,'g')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('title inside1')
 
left, bottom, width, height = 0.6,0.2,0.25,0.25
plt.axes([left,bottom,width,height])
plt.plot(y[::-1],x,'b')
plt.xlabel('x')
plt.ylabel('y')
plt.title('tile inside2')
 
plt.savefig('./')
'''
