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
tex_names = {"t" : "\\theta", "q" : "q_\\mathrm{v}"}

#%%

def tend_prof(dat, var, attrs, cross_dim, loc=None, iloc=None,
             rolling_xwindow=2000, zmax=2000, multiplier=1, sharex=True, xlim=None, **kwargs):
    attrs = dat.attrs.copy()
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

    if kwargs["hue"] == "comp":
        kwargs["palette"] = {"adv_r":"gray", "mean":"tab:blue", "trb_r":"tab:olive", "trb_s":"tab:orange",
                   "trb":"tab:red", "net":"black", "forcing":"violet",
                   "rad":"tab:green", "rad+trb":"tab:brown", "rad_sw" : "green", "rad_lw" : "lime",
                   "pg" : "tab:green", "cor_curv" : "tab:brown", "mp" : "cyan"}
    # c = tuple((mpl.colors.ColorConverter.to_rgb(color) for color in colors))

    datp = dat.copy()
    df = datp.to_dataframe(name="tend").reset_index()

    sns.set_style("whitegrid")
    pgrid = sns.relplot(data=df, kind="line", x="tend", y="hgt", sort_dim="y",
                       facet_kws={"sharex":sharex, "margin_titles":True, "legend_out" : True}, **kwargs)

    if kwargs["hue"] == "comp":
        for ax in pgrid.axes.flatten():
            for l in ax.lines[3:6]:
                l.set_linestyle("--")
                l.set_linewidth(0.8)
        # pgrid.add_legend()

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
    if t not in attrs["description"]:
        t = "flux"

    mult = ""
    if multiplier != 1:
        power = np.log10(1/multiplier)
        if int(power) == power:
            power = int(power)
        mult = "10$^{%s}$ " % power
    pax[-1,middle_column].set_xlabel("$%s$ %s components (%s%s)" % (label, t, mult, attrs["units"]))
    if xlim is not None:
        pgrid.set(xlim=xlim)

    return pgrid

#TODOm: delete?
def scatter_tend_forcing(tend, forcing, var, plot_diff=False, hue="bottom_top", savefig=True, title=None, fname=None, figloc=None, **kwargs):
    if title is None:
        title = fname
    fig, ax, cax = scatter_hue(tend, forcing, plot_diff=plot_diff, hue=hue, title=title,  **kwargs)
    if var in tex_names:
        tex_name = tex_names[var]
    else:
        tex_name = var
    xlabel = "Total ${}$ tendency".format(tex_name)
    ylabel = "Total ${}$ forcing".format(tex_name)
    if plot_diff:
        ylabel += " - " + xlabel
    units = " ({})".format(tools.units_dict_tend[var])
    ax.set_xlabel(xlabel + units)
    ax.set_ylabel(ylabel + units)

    if savefig:
        if figloc is None:
            figloc = tools.figloc
        fig.savefig(figloc + "{}_budget/scatter/{}.png".format(var, fname),dpi=300, bbox_inches="tight")

    return fig

def scatter_hue(dat, ref, plot_diff=False, hue="bottom_top", ignore_missing_hue=False, discrete=False,
                iloc=None, loc=None, savefig=False, close=False, figloc=None, title=None, **kwargs):
    dat = tools.loc_data(dat, loc=loc, iloc=iloc)
    ref = tools.loc_data(ref, loc=loc, iloc=iloc)
    pdat = xr.concat([dat, ref], "concat_dim")

    if plot_diff:#TODOm: change label
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
    for d in [dat, ref]:
        label = ""
        if d.name is not None:
            label = d.name
        elif "description" in d.attrs:
            label = d.description
        if label != "":
            if "units" in d.attrs:
                label += " ({})".format(d.units)
        labels.append(label)
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
    if close:
        plt.show()
        plt.close()

    return fig, ax, cax

def display_ticks(axes):
    plt.subplots_adjust(hspace=0.2)
    for ax in axes.flatten():
        ax.xaxis.set_tick_params(labelbottom=True)