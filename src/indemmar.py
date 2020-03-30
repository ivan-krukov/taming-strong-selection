"""
`indemmar` - context managers for `matplotlib` plots
"""

from contextlib import contextmanager
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

__all__ = ['plot', 'plot_and_legend', 'adaptable_legend_subplot']


def get_all_handles_and_labels(fig):
    """Get handles and labels of every Artist in the figure

    :param fig: Figure to gather the artists and handles from
    :type fig: matplotlib.Figure
    :returns: List of handles, and their labels as a tuple of lists
    :rtype: [matplotlib.Artist], [str]
    """
    handles, labels = [], []
    for ax in fig.get_axes():
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    return handles, labels


def adaptable_legend_subplot(nrow=1, ncol=1, legend_side='right', ratio=10, **figure_kwargs):
    """Creates a layout with separate subplot for the legend
    Parameters
    ----------
    nrow : int
        Number of rows of subplots.
    ncol : int
        Number of columns of subplots.
    legend_side : {'top', 'bottom', 'left', 'right'}
        Which side of the plots should the legend be placed.
    ratio: int
        Ratio of legend subplot to the rest of the subplots.
    **figure_kwargs
        Keyword arguments passed to `matplotlib.Figure` constructor.
    Returns
    -------
    figure: `matplotlib.Figure`
        The figure object
    lax: `matplotlib.Axes`
        The axis of a subplot holding the legend
    ax: `matplotlib.Axes` object or array of Axes objects
        The axes of the subplot to be used for the plot itself.
        A `numpy.array` array of Axes if *nrow* or *ncol* greater than 1.
    """
    fig = plt.figure(constrained_layout=True, **figure_kwargs)
    ax = np.empty((nrow, ncol), dtype=object)  # user-facing axes
    # lax is the legend subplot axes object

    # iterators for all the subplots of the figure (incuding the legend subplot)
    row_start, row_stop = 0, nrow
    col_start, col_stop = 0, ncol

    if legend_side is 'top':
        row_start, row_stop = 1, nrow + 1
        gs = GridSpec(nrow + 1, ncol, fig, height_ratios=[1] + [ratio] * nrow)
        lax = fig.add_subplot(gs[0, :])  # span first row
    elif legend_side is 'bottom':
        gs = GridSpec(nrow + 1, ncol, fig, height_ratios=[ratio] * nrow + [1])
        lax = fig.add_subplot(gs[-1, :])  # span last row
    elif legend_side is 'left':
        col_start, col_stop = 1, ncol + 1
        gs = GridSpec(nrow, ncol + 1, fig, width_ratios=[1] + [ratio] * ncol)
        lax = fig.add_subplot(gs[:, 0])  # span first column
    elif legend_side is 'right':
        gs = GridSpec(nrow, ncol + 1, fig, width_ratios=[ratio] * ncol + [1])
        lax = fig.add_subplot(gs[:, -1])  # span last column
    else:
        raise ArgumentError(
            'Unknown legend_side argument - allowed options are ["top","bottom","left","right"]')

    lax.axis('off')

    # initialize user-facing axes
    for i, row in enumerate(range(row_start, row_stop)):
        for j, col in enumerate(range(col_start, col_stop)):
            ax[i, j] = fig.add_subplot(gs[row, col])

    # flatten if necessary
    ax = ax.flatten() if 1 in ax.shape else ax
    ax = ax[0] if ax.size is 1 else ax

    return fig, lax, ax


@contextmanager
def plot(fname=None, **kwargs):
    fig, ax = plt.subplots(**kwargs)

    try:
        yield (fig, ax)
    finally:
        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname)


@contextmanager
def plot_and_legend(fname=None,
                    legend_title='',
                    legend_side='right',
                    legend_pos=[0.5, 0.5],
                    legend_ncol=1,
                    **figure_kwargs):

    nrow, ncol = figure_kwargs.pop('nrow', 1), figure_kwargs.pop('ncol', 1)
    fig, lax, ax = adaptable_legend_subplot(nrow, ncol, legend_side, **figure_kwargs)

    try:
        yield (fig, ax)
    finally:
        h, l = get_all_handles_and_labels(fig)
        lax.legend(h,
                   l,
                   title=legend_title,
                   bbox_transform=lax.transAxes,
                   bbox_to_anchor=legend_pos,
                   loc='center',
                   ncol=legend_ncol)

        if fname is not None:
            fig.savefig(fname)


if __name__ == '__main__':
    # run examples

    with plot(fname='fig/simple.png') as (fig, ax):
        ax.plot([1, 3, 1, 4, 5, 1, 0])

    with plot_and_legend(ncol=2,
                         fname='fig/legend.png',
                         figsize=(10, 4),
                         legend_side='right',
                         legend_ncol=1) as (fig, ax):

        x = np.linspace(0, np.pi * 2, 100)
        ax[0].plot(x, np.sin(x), label="$y=sin(x)$", color='red')
        ax[1].plot(x, np.cos(x), label="$y=cos(x)$", color='blue')
        fig.suptitle('Periodic functions')
