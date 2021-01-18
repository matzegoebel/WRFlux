#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:38:11 2020

@author: Matthias Göbel
"""
from wrflux import tools
import seaborn as sns
import xarray as xr
xr.set_options(arithmetic_join="exact")
xr.set_options(keep_attrs=True)

import matplotlib.pyplot as plt
import numpy as np
# figloc = tools.figloc
dim_dict = dict(x="U",y="V",bottom_top="W",z="W")
tex_names = {"t": "\\theta", "q": "q_\\mathrm{v}",
             "u": "u", "v": "v", "v": "v"}

#%%

def tend_prof(dat, var, attrs, cross_dim, loc=None, iloc=None, rename=None, extra_xaxis=None,
             rolling_xwindow=2000, zmax=2000, multiplier=1, sharex=True, xlim=None, **kwargs):
    attrs_v = dat.attrs.copy()
    if rolling_xwindow is not None:
        rolling_xwindow = int(rolling_xwindow/attrs["DX"]) + 1
        dat = tools.rolling_mean(dat, cross_dim, rolling_xwindow, periodic=True, center=True)

    dat = dat.where(dat.hgt < zmax)
    if "bottom_top" in dat.dims:
        dat = dat.dropna("bottom_top", "all")
    else:
        dat = dat.dropna("bottom_top_stag", "all")
    dat = multiplier*dat

    dat = tools.loc_data(dat, loc=loc, iloc=iloc)

    cross_dim_u = cross_dim.upper()
    dat = dat.rename({cross_dim : cross_dim_u})
    if rename is not None:
        dat = dat.rename(rename)

    if kwargs["hue"] == "comp":
        kwargs["palette"] = {"adv_r":"gray", "mean":"tab:blue", "trb_r":"tab:olive", "trb_s":"tab:orange",
                   "trb":"tab:red", "net":"black", "forcing":"violet",
                   "rad":"tab:green", "rad+trb":"tab:brown", "rad_sw" : "green", "rad_lw" : "lime",
                   "pg" : "tab:green", "cor_curv" : "tab:brown", "mp" : "cyan"}
    # c = tuple((mpl.colors.ColorConverter.to_rgb(color) for color in colors))

    datp = dat.copy()
    df = datp.to_dataframe(name="tend").reset_index()

    sns.set_style("whitegrid")
    if extra_xaxis is not None:
        #exclude locations in extra_xaxis from plot but not from legend
        df = df.copy()
        for key, val in extra_xaxis.items():
            df["tend"] = df["tend"].where(df[key] != val, np.inf)
    pgrid = sns.relplot(data=df, kind="line", x="tend", y="hgt", sort_dim="y",
                       facet_kws={"sharex":sharex, "margin_titles":True, "legend_out" : True}, **kwargs)


    pgrid.set_xlabels("")
    pgrid.set_ylabels("height above ground (m)")
    pax = pgrid.axes

    if sharex == "row":
        display_ticks(pax)
    middle_column =  int((len(pax[0])-1)/2)
    label = var
    if var in tex_names:
        label = tex_names[var]

    t = "tendency"
    if t not in attrs_v["description"]:
        t = "flux"

    mult = ""
    if multiplier != 1:
        power = np.log10(1/multiplier)
        if int(power) == power:
            power = int(power)
        mult = "10$^{%s}$ " % power
    pax[-1,middle_column].set_xlabel("$%s$ %s components (%s%s)" % (label, t, mult, attrs_v["units"]))

    pax_flat = pax.flatten()
    if extra_xaxis is not None:
        #plot locations in extra_xaxis with separate x-axis for each subplot
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
        pax2[-1,middle_column].set_xlabel("$%s$ %s components sum (%s%s)" % (label, t, mult, attrs_v["description"]))

    if xlim is not None:
        pgrid.set(xlim=xlim)

    return pgrid

def scatter_hue(dat, ref, plot_diff=False, hue="bottom_top", ignore_missing_hue=False, discrete=False,
                iloc=None, loc=None, savefig=False, close=False, figloc=None, title=None, **kwargs):
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

    #set color
    cmap = "cool"
    if ("bottom_top" in hue) and (not discrete):
        color = -pdatf[hue]
    elif (hue == "Time") and (not discrete):
        color = pdatf["hue"]
    else:
        color = pdatf[hue]
        try:
            color.astype(int) #check if hue is numeric
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

    for i in [0,1]:
        pdat = pdat.where(~pdat[i].isnull())
    if not plot_diff:
        minmax = [pdat.min(), pdat.max()]
        dist = minmax[1] - minmax[0]
        minmax[0] -= 0.03*dist
        minmax[1] += 0.03*dist
        plt.plot(minmax, minmax, c="k")
        ax.set_xlim(minmax)
        ax.set_ylim(minmax)

    #colorbar
    cax = fig.add_axes([0.92,0.125,0.05,.75], frameon=True)
    cax.set_yticks([])
    cax.set_xticks([])
    clabel = hue
    if "bottom_top" in hue:
        clabel = "$\eta$"
    if ("bottom_top" in hue) and (not discrete):
        cb = plt.colorbar(p,cax=cax,label=clabel)
        cb.set_ticks(np.arange(-0.8,-0.2,0.2))
        cb.set_ticklabels(np.linspace(0.8,0.2,4).round(1))
    else:
        cb = plt.colorbar(p,cax=cax,label=clabel)
        if discrete:
            if n_hue > 1:
                d = (n_hue-1)/n_hue
                cb.set_ticks(np.arange(d/2, n_hue-1, d))
            else:
                cb.set_ticks([0])

            cb.set_ticklabels(pdat[hue].values)

    #error labels
    err = abs(dat - ref)
    rmse = (err**2).mean().values**0.5
    ns = tools.nse(dat, ref)
    ax.text(0.74,0.07,"RMSE={0:.2E}\nNSE={1:.7f}".format(rmse, ns.values),
            horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
    if title is not None:
        fig.suptitle(title)

    # if savefig:
    #     if figloc is None:
    #         figloc = "~/"
    #     fig.savefig(figloc + "{}_budget/scatter/{}.png".format(var, fname),dpi=300, bbox_inches="tight")
    plt.show()
    if close:
        plt.close()

    return fig, ax, cax

def display_ticks(axes):
    plt.subplots_adjust(hspace=0.2)
    for ax in axes.flatten():
        ax.xaxis.set_tick_params(labelbottom=True)