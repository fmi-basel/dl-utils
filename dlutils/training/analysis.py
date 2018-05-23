from scipy.ndimage.filters import gaussian_filter1d

import numpy as np
import matplotlib.pyplot as plt


def plot_log(path, mean_sigma=3, **kwargs):
    '''plot a training log.

    Parameters
    ----------
    path : string
        path to logfile.
    mean_sigma : float, optional
        sigma to of Gaussian average for mean estimate.

    '''
    data = np.genfromtxt(path, delimiter=',', names=True)
    _, axarr = plt.subplots(
        1,
        len([
            name for name in data.dtype.names
            if 'val_' not in name and name != 'epoch'
        ]),
        sharex=True,
        sharey=True,
        **kwargs)

    def plot_with_mean(x, y, label=None):
        y_hat = gaussian_filter1d(y, sigma=mean_sigma, mode='nearest')
        plot_handle = ax.plot(x, y, alpha=0.5)
        ax.plot(
            x,
            y_hat,
            color=plot_handle[-1].get_color(),
            linewidth=3,
            label=label)

    for ax, key in zip(axarr, (key for key in data.dtype.names
                               if key != 'epoch' and 'val' not in key)):
        plot_with_mean(data['epoch'], data[key])
        if 'val_' + key in data.dtype.names:
            plot_with_mean(data['epoch'], data['val_' + key])
        ax.set_ylabel(key)
        ax.set_yscale('log')
