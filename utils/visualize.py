import contextlib
import numpy as np

from PIL import Image

import torch
import torchvision.utils as vutils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()
#sns.set_style('whitegrid')
#sns.set_palette('colorblind')


# for color palette. https://stackoverflow.com/a/49557127
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

with temp_seed(0):
    COLOR_PALETTE = sns.color_palette(n_colors=8)+[tuple(np.random.rand(3,).tolist()) for i in range(992)]


def convert_npimage_torchimage(image):
    image = image.astype('float') / 255.
    return torch.transpose(torch.transpose(torch.from_numpy(image), 0, 2), 1, 2).float()

#def clamp_image(grid):
#    """
#    https://pytorch.org/docs/stable/_modules/torchvision/utils.html#save_image
#    """
#    grid = grid.detach().clone()
#    return grid.mul(255).add_(0.5).clamp_(0, 255).div_(255)

def get_scatter_plot(data, labels=None, n_classes=None, num_samples=1000, xlim=None, ylim=None, alpha=1.0, use_grid=True):
    '''
    data   : 2d points, batch_size x data_dim (numpy array)
    labels : labels, batch_size (numpy array)
    '''
    batch_size, data_dim = data.shape
    num_samples = min(num_samples, batch_size)
    if labels is None:
        labels = np.zeros(batch_size, dtype=np.int)
    if n_classes is None:
        n_classes = len(np.unique(labels))

    # sub-samples
    if num_samples != batch_size:
        indices = np.random.permutation(batch_size)
        data = data[indices[:num_samples]]
        labels = labels[indices[:num_samples]]

    # init config
    #palette = sns.color_palette(n_colors=n_classes)
    #palette = [palette[i] for i in np.unique(labels)]
    palette = [COLOR_PALETTE[i%len(COLOR_PALETTE)] for i in np.unique(labels)]

    # plot
    fig, ax = plt.subplots(figsize=(5, 5))
    data = {'x': data[:, 0],
            'y': data[:, 1],
            'class': labels}
    sns.scatterplot(x='x', y='y', hue='class', data=data, palette=palette, alpha=alpha, linewidth=0)
    ax.get_legend().remove()
    ax.grid(False)

    # set config
    if xlim is not None:
        plt.xlim((-xlim, xlim))
    if ylim is not None:
        plt.ylim((-ylim, ylim))

    # use grid
    if use_grid:
        plt.xticks(np.arange(-xlim, xlim+1, step=1))
        plt.yticks(np.arange(-ylim, ylim+1, step=1))
    else:
        plt.xticks([])
        plt.yticks([])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    # tight
    plt.tight_layout()

    # draw to canvas
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # close figure
    plt.close()
    return image

def get_imshow_plot(image:torch.Tensor, figsize=(5, 5), plot=False):
    image = image[0].permute(1, 2, 0).numpy()
    #image = image*255.astype(np.uint8)

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image)
    ax.grid(False)
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False, # labels along the bottom edge are off
        )

    # tight
    plt.tight_layout()

    if plot:
        plt.show()
    else:
        # draw to canvas
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        #image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # close figure
        plt.close()
        return image

def get_2d_histogram_plot(data:torch.Tensor, val=5, num=256, use_grid=False, origin='lower', figsize=(5, 5)):
    xsz = list(data.shape)
    batch_size, num_points, xdim = xsz[0], xsz[1], np.prod(xsz[2:])
    assert num_points == 1 and xdim == 2
    data = data.view(batch_size, xdim).cpu().numpy()

    xmin = -val
    xmax = val
    ymin = -val
    ymax = val

    # get data
    x = data[:, 0]
    y = data[:, 1]

    # get histogram
    heatmap, xedges, yedges = np.histogram2d(x, y, range=[[xmin, xmax], [ymin, ymax]], bins=num)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(heatmap.T, extent=extent, cmap='jet', origin=origin)
    ax.grid(False)
    if use_grid:
        plt.xticks(np.arange(-val, val+1, step=1))
        plt.yticks(np.arange(-val, val+1, step=1))
    else:
        plt.xticks([])
        plt.yticks([])

    # tight
    plt.tight_layout()

    if plot:
        plt.show()
    else:
        # draw to canvas
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # close figure
        plt.close()
        return image

def get_1d_plot(x:torch.Tensor,
                v:torch.Tensor,
                num_plot=1,
                xmin:float=-np.pi,
                xmax:float=np.pi,
                ymin:float=-np.pi/2.,
                ymax:float=np.pi/2.,
                sort:bool=True,
                #use_grid=False,
                origin='lower',
                figsize=(5, 5),
                color=None,
                alpha=1.0,
                plot=False,
                ):
    xsz = list(x.shape)
    batch_size, num_points, xdim = xsz[0], xsz[1], np.prod(xsz[2:])
    vsz = list(v.shape)
    b, n, vdim = vsz[0], vsz[1], np.prod(vsz[2:])
    assert b == batch_size
    assert (n == num_points and vdim == 1 and xdim == 1) or \
           (vdim == xdim and n == 1 and num_points == 1)

    x = x.view(batch_size, num_points*xdim)
    v = v.view(batch_size, num_points*vdim)

    if sort:
        v, indices = torch.sort(v, dim=1)
        x = x[torch.arange(batch_size)[:,None], indices]

    fig = plt.figure(figsize=figsize)
    for i in range(min(batch_size, num_plot)):
        if color is not None:
            plt.plot(v[i].cpu().numpy(), x[i].cpu().numpy(), alpha=alpha, color=color)
        else:
            plt.plot(v[i].cpu().numpy(), x[i].cpu().numpy(), alpha=alpha)
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    ax = plt.gca()
    ax.grid(False)

    # tight
    plt.tight_layout()

    if plot:
        plt.show()
    else:
        # draw to canvas
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # close figure
        plt.close()
        return image


#######################################################################################
def preproc_nodes(pos_x, eps=0.):
    num_points, pos_dim = pos_x.shape
    if pos_dim == 1:
        pos_y = torch.zeros(num_points, 2).to(pos_x.device)
        pos_y[:,:1] = pos_x
        pos_y[:,1:] = pos_x + eps
    elif pos_dim == 2:
        pos_y = pos_x
    else:
        raise NotImplementedError
    return pos_y

def vis_edges(pos_src, pos_dst, edge_index, scale=1.0, head_scale=None, eps=0.05, **kwargs):
    if head_scale is None:
        head_scale = scale
    pos_src = preproc_nodes(pos_src, eps=0.)
    pos_dst = preproc_nodes(pos_dst, eps=eps)
    for (src_, dst_) in edge_index.t().tolist():
            src = (scale*pos_src[src_]).tolist()
            dst = (scale*pos_dst[dst_]).tolist()
            plt.arrow(src[0], src[1], dst[0]-src[0], dst[1]-src[1],
                    head_length=head_scale*0.075,
                    head_width=head_scale*0.075,
                    shape='full',
                    length_includes_head=True,
                    color='lightgray',
                    **kwargs,
                    )

def vis_nodes(pos_x, color, s=50, zorder=1000, scale=1.0, eps=0., **kwargs):
    pos_x = preproc_nodes(pos_x, eps=eps)
    plt.scatter(scale*pos_x[:, 0], scale*pos_x[:, 1], s=s, zorder=zorder, color=color, **kwargs)

def vis_circles(pos_x, radius, color='r', s=50, zorder=1000, scale=1.0, eps=0.):
    pos_x = preproc_nodes(pos_x, eps=eps)
    for i in range(pos_x.shape[0]):
        pos_x_ = scale*pos_x[i]
        circle = plt.Circle((pos_x_[0].item(), pos_x_[1].item()), scale*radius, color=color, fill=False)
        ax = plt.gca()
        ax.add_patch(circle)


#######################################################################################
def get_grid_image(x, nrow=8, pad_value=0, padding=2, normalize=True, to_numpy=False):
    output = vutils.make_grid(x[:nrow**2], nrow=nrow, normalize=normalize, scale_each=True, pad_value=pad_value, padding=padding)
    if to_numpy:
        return (output.permute(1, 2, 0).numpy()*255).astype(np.uint8)
    else:
        return output

def get_grid_image_from_xv(x, v, xv_to_img, nrow=8, pad_value=0, padding=2, to_numpy=False):
    '''
    input : b x c x h x w (where h = w)
    '''
    output = xv_to_img(x[:nrow**2], v[:nrow**2]).clone().cpu()
    output = vutils.make_grid(output, nrow=nrow, normalize=True, scale_each=True, pad_value=pad_value, padding=padding)
    if to_numpy:
        return (output.permute(1, 2, 0).numpy()*255).astype(np.uint8)
    else:
        return output

def _get_imgs_selected_inds(xs:torch.Tensor, inds:list, plot_func=get_2d_histogram_plot, **kwargs):
    imgs = []
    for ind in inds:
        imgs += [plot_func(xs[ind], **kwargs)]
    imgs = np.concatenate(imgs, axis=1)
    return imgs

def _plot_imgs_selected_inds(imgs:np.array, inds:list, lmbd=0., use_xticks=True, use_yticks=True, plot=False, **kwargs):
    height, width, _ = imgs.shape
    height_per_img = width_per_img = height
    figwidth = 25
    fontsize = 15
    if use_xticks:
        xticks = [0.5*width_per_img + width_per_img*i for i in range(len(inds))]
        xticklabels = [r'$i={:d}$'.format(ind+1) for ind in inds]
    else:
        xticks, xticklabels = [], []
    if use_yticks:
        yticks = [0.5*height_per_img]
        yticklabels = [r'$\lambda={:.2g}$'.format(lmbd)]
    else:
        yticks, yticklabels = [], []

    fig = plt.figure(figsize=(figwidth, figwidth*height/width))
    ax = fig.add_subplot(111)
    ax.imshow(imgs)
    ax.grid(False)
    axis_color = 'white' #'white'
    ax.spines['bottom'].set_color(axis_color)
    ax.spines['top'].set_color(axis_color)
    ax.spines['left'].set_color(axis_color)
    ax.spines['right'].set_color(axis_color)
    ax.tick_params(axis='x', colors=axis_color)
    ax.tick_params(axis='y', colors=axis_color)
    plt.xticks(xticks, xticklabels, color='black', fontsize=fontsize)
    plt.yticks(yticks, yticklabels, color='black', fontsize=fontsize)

    # tight
    fig.tight_layout(pad=0)

    # plot
    if plot:
        plt.show()
    else:
        # draw to canvas
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # close figure
        plt.close()
        return image

def plot_selected_inds(
        xs:torch.Tensor,
        inds:list,
        lmbd=0.,
        use_xticks=True,
        use_yticks=True,
        plot_func=get_2d_histogram_plot,
        plot=False,
        **kwargs):
    imgs = _get_imgs_selected_inds(xs=xs, inds=inds, plot_func=plot_func, **kwargs)
    return _plot_imgs_selected_inds(imgs=imgs, inds=inds, lmbd=lmbd, plot=plot, **kwargs)




def plot_scatter(x, v, nrows=2, ncols=2, figsize=(9,8)):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for b, ax in enumerate(axes.flat):
        x0, v0 = x[b], v[b]
        im = ax.scatter(v0[:,1], v0[:,0], c=x0[:,0], vmin=-0.1, vmax=0.5, cmap='viridis')
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.invert_yaxis()
        ax.axis('equal')
        ax.set_facecolor((1.0, 1.0, 1.0))
    fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.02)
    plt.show()

def plot_contourf(x, v, img_height, nrows=2, ncols=2, figsize=(9,8), plot=True):
    # print('plot', x.shape)
    # print('plot', v.shape)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes_ = [axes]
    else:
        axes_ = axes.flat
    for b, ax in enumerate(axes_):
        x0, v0 = x[b].cpu(), v[b].cpu()
        # print('x0', x0.shape)
        # print('v0', v0.shape)
        im = ax.scatter(v0[:,1], v0[:,0], c=x0[:,0], vmin=-0.1, vmax=0.5, cmap='viridis')
        ax.contourf(
            v0.reshape((img_height,img_height,2))[:,:,1],
            v0.reshape((img_height,img_height,2))[:,:,0],
            x0.reshape((img_height,img_height)),
            alpha=0.8,
            cmap='viridis',
        )
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.invert_yaxis()
        ax.axis('equal')
        ax.grid(False)
        ax.tick_params(
            axis='both',       # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False, # labels along the bottom edge are off
            )
        ax.set_facecolor((1.0, 1.0, 1.0))

    # tight
    plt.tight_layout()

    if plot:
        fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.02)
        plt.show()
    else:
        # draw to canvas
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # close figure
        plt.close()
        return image

def get_contourf_plot(x, v, img_height, figsize=(4,4)):
    return plot_contourf(x=x, v=v, img_height=img_height, nrows=1, ncols=1, figsize=figsize, plot=False)
