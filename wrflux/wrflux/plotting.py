#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for plotting WRFlux output data:
scatter plots

@author: Matthias Göbel
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from wrflux import tools
import xarray as xr
from pathlib import Path
xr.set_options(arithmetic_join="exact")
xr.set_options(keep_attrs=True)
dim_dict = dict(x="U", y="V", bottom_top="W", z="W")
tex_names = {"t": "\\theta", "q": "q_\\mathrm{v}",
             "u": "u", "v": "v", "v": "v"}

# %%


def scatter_hue(dat, ref, plot_diff=False, hue="bottom_top", ignore_missing_hue=False,
                discrete=False, iloc=None, loc=None, savefig=False, fname=None, figloc=None,
                close=False, title=None, **kwargs):
    """Scatter plot of dat vs. ref with coloring based on hue variable.


    Parameters
    ----------
    dat : xarray DataArray
        Data plotted on y-axis.
    ref : xarray DataArray
        Reference data plotted on x-axis.
    plot_diff : bool, optional
        Plot the difference dat-ref against ref. The default is False.
    hue : str, optional
        Hue variable. All xarray dimensions are allowed. "_stag" is automatically appended,
        if necessary. The default is "bottom_top".
    ignore_missing_hue : bool, optional
        If hue variable is not available, use default instead of raising an exception.
        The default is False.
    discrete : bool, optional
        Use discrete color map to facilitate discrimination of close values. The default is False.
    loc : dict, optional
        Mapping for label based indexing before plotting. The default is None.
    iloc : dict, optional
        Mapping for integer-location based indexing before plotting. The default is None.
    savefig : bool, optional
        Save figure to disk. The default is False.
    fname : str, optional
        File name of plot if savefig=True. If no file type extension is included, use png.
        The default is None.
    figloc : str or path-like, optional
        Directory to save plot in. Defaults to the directory of this script.
    close : bool, optional
        Close the figure after creation. The default is False.
    title : str, optional
        Title of the plot. The default is None.
    **kwargs :
        keyword argument passed to plt.scatter.

    Returns
    -------
    fig : matplotlib figure
    ax : matplotlib axes
    cax : matplotlib axes
        Colorbar axes.

    """
    dat = tools.loc_data(dat, loc=loc, iloc=iloc)
    ref = tools.loc_data(ref, loc=loc, iloc=iloc)
    pdat = xr.concat([dat, ref], "concat_dim")

    if plot_diff:
        pdat[0] = dat - ref

    if ignore_missing_hue:
        if ((hue not in pdat.dims) and (hue + "_stag" not in pdat.dims)):
            hue = "bottom_top"
            discrete = False
    if (hue not in pdat.dims) and (hue + "_stag" in pdat.dims):
        hue = hue + "_stag"

    # create integer hue variable to allow non-numeric hue variables
    n_hue = len(pdat[hue])
    hue_int = np.arange(n_hue)
    pdat = pdat.assign_coords(hue=(hue, hue_int))
    pdatf = pdat[0].stack(s=pdat[0].dims)

    # set color
    cmap = "gnuplot"
    if ("bottom_top" in hue) and (not discrete):
        color = -pdatf[hue]
    elif (hue == "Time") and (not discrete):
        # use integer hue variable to prevent error
        color = pdatf["hue"]
    else:
        color = pdatf[hue]
        try:
            color.astype(int)  # check if hue is numeric
        except ValueError:
            discrete = True
        if discrete:
            cmap = plt.get_cmap(cmap, n_hue)
            discrete = True
            # use integer hue variable to prevent error
            color = pdatf["hue"]

    kwargs.setdefault("cmap", cmap)

    fig, ax = plt.subplots()
    kwargs.setdefault("s", 10)
    p = plt.scatter(pdat[1], pdat[0], c=color.values, **kwargs)

    # set x and y labels
    labels = []
    for d in [ref, dat]:
        label = ""
        if d.name is not None:
            label = d.name
        elif "description" in d.attrs:
            label = d.description
        labels.append(label)
    if plot_diff and (labels[0] != "") and (labels[1] != ""):
        labels[1] = "{} - {}".format(labels[1], labels[0])
    # add units
    for i, d in enumerate([ref, dat]):
        if (labels[i] != "") and ("units" in d.attrs):
            labels[i] += " ({})".format(d.units)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    for i in [0, 1]:
        pdat = pdat.where(~pdat[i].isnull())
    # set axis limits equal for x and y axis
    if not plot_diff:
        minmax = [pdat.min(), pdat.max()]
        # tak full range of data increased by 3%
        dist = minmax[1] - minmax[0]
        minmax[0] -= 0.03 * dist
        minmax[1] += 0.03 * dist
        plt.plot(minmax, minmax, c="gray", label="1:1")
        plt.legend()
        ax.set_xlim(minmax)
        ax.set_ylim(minmax)

    # colorbar
    cax = fig.add_axes([0.92, 0.125, 0.05, .75], frameon=True)
    cax.set_yticks([])
    cax.set_xticks([])
    clabel = hue
    if "bottom_top" in hue:
        clabel = "$\eta$"
    if ("bottom_top" in hue) and (not discrete):
        cb = plt.colorbar(p, cax=cax, label=clabel)
        # highest value must be at bottom
        cb.set_ticks(np.arange(-0.8, -0.2, 0.2))
        cb.set_ticklabels(np.linspace(0.8, 0.2, 4).round(1))
    else:
        cb = plt.colorbar(p, cax=cax, label=clabel)
        if discrete:
            # set ticks for all hue values
            if n_hue > 1:
                d = (n_hue - 1) / n_hue
                cb.set_ticks(np.arange(d / 2, n_hue - 1, d))
            else:
                cb.set_ticks([0])
            cb.set_ticklabels(pdat[hue].values)

    # labels for error stats
    err = abs(dat - ref)
    rmse = (err**2).mean().values**0.5
    ns = tools.R2(dat, ref)
    ax.text(0.74, 0.07, "RMSE={0:.2E}\nR$^2$={1:.7f}".format(rmse, ns.values),
            horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
    if title is not None:
        fig.suptitle(title)

    if savefig:
        if figloc is None:
            figloc = Path(__file__).parent
        else:
            os.makedirs(figloc, exist_ok=True)
        if fname is None:
            fname = "scatter"
        fpath = Path(figloc) / fname
        try:
            fig.savefig(fpath, dpi=300, bbox_inches="tight")
        except ValueError:
            fig.savefig(str(fpath) + ".png", dpi=300, bbox_inches="tight")

    plt.show()
    if close:
        plt.close()

    return fig, ax, cax
