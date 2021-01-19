#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for plotting WRFlux output data:
tendency profiles and scatter plots

@author: Matthias Göbel
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from wrflux import tools
import seaborn as sns
import xarray as xr
xr.set_options(arithmetic_join="exact")
xr.set_options(keep_attrs=True)
dim_dict = dict(x="U", y="V", bottom_top="W", z="W")
tex_names = {"t": "\\theta", "q": "q_\\mathrm{v}",
             "u": "u", "v": "v", "v": "v"}

# %%

# TODOm possible with standard seaborn?


def tend_prof(dat, var, loc=None, iloc=None, rename=None, extra_xaxis=None,
              rolling_xwindow=None, rolling_dim=None, zmax=2000, multiplier=1,
              sharex=True, xlim=None, **kwargs):
    """Plot tendency profiles for variable var using seaborn's relplot with all its
    options for  grouping by style, hue, and size and faceting.


    Parameters
    ----------
    dat : xarray DataArray
        Tendency components for variable var.
    var : str
        Variable to plot.
    loc : dict, optional
        Mapping for label based indexing before plotting. The default is None.
    iloc : dict, optional
        Mapping for integer-location based indexing before plotting. The default is None.
    rename : dict, optional
        Rename coordinates. The default is None.
    extra_xaxis : dict, optional
        Mapping from coordinates to keys for which to draw separate x-axis in each subplot.
        The default is None.
    rolling_xwindow : float, optional
        If given, do rolling average in dimension "rolling_dim" with the given window size.
        The default is None.
    rolling_dim : str, optional
        Dimension for rolling mean. The default is None.
    zmax : float, optional
        Upper limit of y-axis: height (m). The default is 2000.
    multiplier : float, optional
        Multiply data with this factor. The default is 1.
    sharex : bool, optional
        Share x-axis among subplots. The default is True.
    xlim : tuple of float (xmin, xmax), optional
        Set lower and upper limit of x-axis . The default is None.
    **kwargs :
        Keyword arguments passed to seaborn's relplot.

    Returns
    -------
    pgrid : FacetGrid

    """
    attrs_v = dat.attrs.copy()
    if rolling_xwindow is not None:
        dx = dat[rolling_dim][-1] - dat[rolling_dim][-2]
        rolling_xwindow = int(rolling_xwindow / dx) + 1
        dat = tools.rolling_mean(dat, rolling_dim, rolling_xwindow, periodic=True, center=True)

    dat = dat.where(dat.hgt < zmax)
    if "bottom_top" in dat.dims:
        dat = dat.dropna("bottom_top", "all")
    else:
        dat = dat.dropna("bottom_top_stag", "all")
    dat = multiplier * dat

    dat = tools.loc_data(dat, loc=loc, iloc=iloc)

    for d in tools.xy:
        if d in dat.coords:
            dat = dat.rename({d: d.upper()})

    if rename is not None:
        dat = dat.rename(rename)

    if kwargs["hue"] == "comp":
        kwargs["palette"] = {"adv_r": "gray", "mean": "tab:blue", "trb_r": "tab:olive",
                             "trb_s": "tab:orange", "trb": "tab:red", "net": "black",
                             "forcing": "violet", "rad": "tab:green", "rad+trb": "tab:brown",
                             "rad_sw": "green", "rad_lw": "lime",
                             "pg": "tab:green", "cor_curv": "tab:brown", "mp": "cyan"}

    datp = dat.copy()
    df = datp.to_dataframe(name="tend").reset_index()

    sns.set_style("whitegrid")
    if extra_xaxis is not None:
        # exclude locations in extra_xaxis from plot but not from legend
        df = df.copy()
        for key, val in extra_xaxis.items():
            df["tend"] = df["tend"].where(df[key] != val, np.inf)
    pgrid = sns.relplot(data=df, kind="line", x="tend", y="hgt", sort_dim="y",
                        facet_kws={"sharex": sharex, "margin_titles": True, "legend_out": True},
                        **kwargs)

    pgrid.set_xlabels("")
    pgrid.set_ylabels("height above ground (m)")
    pax = pgrid.axes

    if sharex == "row":
        display_ticks(pax)
    middle_column = int((len(pax[0]) - 1) / 2)
    label = var
    if var in tex_names:
        label = tex_names[var]

    t = "tendency"
    if t not in attrs_v["description"]:
        t = "flux"

    mult = ""
    if multiplier != 1:
        power = np.log10(1 / multiplier)
        if int(power) == power:
            power = int(power)
        mult = "10$^{%s}$ " % power
    pax[-1, middle_column].set_xlabel("$%s$ %s components (%s%s)" % (label, t, mult,
                                                                     attrs_v["units"]))
    pax_flat = pax.flatten()
    if extra_xaxis is not None:
        # plot locations in extra_xaxis with separate x-axis for each subplot
        df = datp.loc[extra_xaxis].to_dataframe(name="tend").reset_index()
        kwargs_sub = {}
        for k in kwargs:
            if k not in ["aspect", "height", "col", "row"]:
                kwargs_sub[k] = kwargs[k]
        sns.set_style("ticks")
        pax2 = []
        for ax in pax_flat:
            ax2 = ax.twiny()
            pax2.append(ax2)
            sns.lineplot(data=df, legend=False, ax=ax2, x="tend", y="hgt", sort_dim="y", **kwargs_sub)
        pax_flat = [*pax_flat, *pax2]
        pax2 = np.array(pax2).reshape(pax.shape)
        pax2[-1, middle_column].set_xlabel("$%s$ %s components sum (%s%s)" % (label, t, mult,
                                                                              attrs_v["description"]))

    if xlim is not None:
        pgrid.set(xlim=xlim)

    return pgrid


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
    figloc : str, optional
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

    n_hue = len(pdat[hue])
    hue_int = np.arange(n_hue)
    pdat = pdat.assign_coords(hue=(hue, hue_int))
    pdatf = pdat[0].stack(s=pdat[0].dims)

    # set color
    cmap = "cool"
    if ("bottom_top" in hue) and (not discrete):
        color = -pdatf[hue]
    elif (hue == "Time") and (not discrete):
        color = pdatf["hue"]
    else:
        color = pdatf[hue]
        try:
            color.astype(int)  # check if hue is numeric
        except:
            discrete = True
        if discrete:
            cmap = plt.get_cmap("tab20", n_hue)
            if n_hue > 20:
                raise ValueError("Too many different hue values for cmap tab20!")
            discrete = True
            color = pdatf["hue"]

    kwargs.setdefault("cmap", cmap)

    fig, ax = plt.subplots()
    kwargs.setdefault("s", 10)
    p = plt.scatter(pdat[1], pdat[0], c=color.values, **kwargs)
    labels = []
    for d in [ref, dat]:
        label = ""
        if d.name is not None:
            label = d.name
        elif "description" in d.attrs:
            label = d.description
        labels.append(label)

    if (labels[0] != "") and (labels[1] != "") and plot_diff:
        labels[1] = "{} - {}".format(labels[1], labels[0])
    for i, d in enumerate([ref, dat]):
        if (labels[i] != "") and ("units" in d.attrs):
            labels[i] += " ({})".format(d.units)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    for i in [0, 1]:
        pdat = pdat.where(~pdat[i].isnull())
    if not plot_diff:
        minmax = [pdat.min(), pdat.max()]
        dist = minmax[1] - minmax[0]
        minmax[0] -= 0.03 * dist
        minmax[1] += 0.03 * dist
        plt.plot(minmax, minmax, c="k")
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
        cb.set_ticks(np.arange(-0.8, -0.2, 0.2))
        cb.set_ticklabels(np.linspace(0.8, 0.2, 4).round(1))
    else:
        cb = plt.colorbar(p, cax=cax, label=clabel)
        if discrete:
            if n_hue > 1:
                d = (n_hue - 1) / n_hue
                cb.set_ticks(np.arange(d / 2, n_hue - 1, d))
            else:
                cb.set_ticks([0])

            cb.set_ticklabels(pdat[hue].values)

    # error labels
    err = abs(dat - ref)
    rmse = (err**2).mean().values**0.5
    ns = tools.nse(dat, ref)
    ax.text(0.74, 0.07, "RMSE={0:.2E}\nNSE={1:.7f}".format(rmse, ns.values),
            horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
    if title is not None:
        fig.suptitle(title)

    if savefig:
        if figloc is None:
            figloc = os.path.abspath(os.path.dirname(__file__))
        else:
            os.makedirs(figloc, exist_ok=True)
        if fname is None:
            fname = "scatter"
        if "." not in fname:
            fname += ".png"
        fig.savefig(figloc + "/" + fname, dpi=300, bbox_inches="tight")
    plt.show()
    if close:
        plt.close()

    return fig, ax, cax


def display_ticks(axes):
    plt.subplots_adjust(hspace=0.2)
    for ax in axes.flatten():
        ax.xaxis.set_tick_params(labelbottom=True)
