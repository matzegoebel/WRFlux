#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:54:43 2020

Calculate the WRF budget terms for theta or qv under different assumptions.
Scatter and profile plots of the individual terms and sums.

@author: c7071088
"""

import os
import tools
from tools import tex_names, units_dict
from run_wrf import misc_tools
import metpy.constants as metconst
import seaborn as sns
import wrf
import xarray as xr
xr.set_options(arithmetic_join="exact")
xr.set_options(keep_attrs=True)

import matplotlib.pyplot as plt
import numpy as np
import sys
figloc = tools.figloc

os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"

print(wrf.omp_enabled())
wrf.omp_set_num_threads(8)

#TODO: find remaining error sources: -g factor, dzdnw or else in sgs
#TODO: plot with x averaging -> mean and x tendencies in flat get small, problem when only averaging over y? Larger domain?
#-> compare different averaging domains, how large does domain in y need to be to get reasonable x tendencies (0 in flat, something in mnt)
#-> include mean wind: non-stationary cells, diagonal wind?

#TODO: go through step by step to check code
#%%settings

conf = None
config="test.config_test_fluxes_real"
exec("import run_wrf.configs.{} as conf".format(config))

wrf_build = "{}/WRF".format(conf.build_path)
outpath = conf.outpath + conf.outdir + "/"

comb_i = 0 #index of the run in the param_grid to analyse
# avg_dim = ["x","y"] #spatial averaging dimension, x and/or y
avg_dim = "y" #spatial averaging dimension, x and/or y
hor_avg = False #average over avg_dim
t_avg = False #average over time
t_avg_interval = 4 #size of the time averaging window

var = "w" #budget variable, q or th


savefig = False
#%%load and optionally average data

VAR = var.upper()
dim_dict = dict(x="U",y="V",bottom_top="W",z="W")
xy = ["x", "y"]
XY = ["X", "Y"]

vel = [dim_dict[d] for d in xy]
xy_v = list(zip(xy, vel))

#create output ID for current configuration
IDi, IDi_d = misc_tools.output_id_from_config(conf.param_combs[comb_i], conf.param_grid, conf.param_names, conf.runID)
IDi_n = list(IDi_d.items())
IDi_n = "_".join(["_".join([str(i) for i in pair]) for pair in IDi_n])
# namelist = "{}/WRF_{}_0/namelist.input".format(conf.run_path, IDi)
# namelist = namelist_to_dict(namelist, build_path=wrf_build, registries=conf.registries)
print("\n\n\n\n Simulation: {}\n\n".format(dict(IDi_d)))

dat_inst = tools.open_dataset(outpath + "/fastout_{}_0".format(IDi), del_attrs=False)
dat_mean = tools.open_dataset(outpath + "/slowout_{}_0".format(IDi))

dat_mean, dat_inst, total_tend, sgs, sources, attrs, grid, dim_stag, stagger_const, cyclic, mapfac, dzdd, dzdd_s \
 = tools.prepare(dat_mean, dat_inst, var, t_avg=t_avg, t_avg_interval=t_avg_interval)

for f in ["RU", "RV"]:
    #divide total fluxes by gravity
    f = f + VAR + "_TOT_MEAN_2ND"
    if f in dat_mean:
        dat_mean[f] = dat_mean[f]/(-9.81)

#%%calc fluxes and tendencies

IDc_single = "cartesian correct recalc_w" #single setting to test
IDc_single = [[], "0"]
# IDc_single = None
plot_diff = False #plot difference between forcing and tendency against tendency
cut_boundaries = False #cut lateral, lower and upper boundaries for plotting
hue = "eta"

flux = None
adv = xr.Dataset()
forcing = xr.Dataset()
keys = ["cartesian","correct","recalc_w","force_2nd_adv"] #available settings
short_names = {"2nd" : "force_2nd_adv", "corr" : "correct"} #abbreviations for settings
ID_rename = {"0" : "3rd+5th"} #abbreviations for combs
base_setting = ["cartesian"]#["cartesian"] #settings for all combs
#all settings to test (if IDc_single is None)
combs = [
          # [[], "0"],
          ["2nd"],
          # ["recalc_w", "2nd"],
          ["correct", "2nd"],
          ["correct", "recalc_w", "2nd"],
          ["correct", "recalc_w"]

         ]
# base_setting = []#["cartesian"] #settings for all combs
# combs = [["correct", "recalc_w", "cartesian"], [[],"0"]]

IDcs = []

for i,comb in enumerate(combs):
    if type(comb[0]) not in [tuple, list]:
        combs[i] = [comb, " ".join(comb)]
    IDcs.append(combs[i][1])

if IDc_single is not None:
    base_setting = []
    if type(IDc_single) in (list,tuple):
        IDcs = [IDc_single[1]]
        combs = [IDc_single]
    else:
        try:
            combs = [combs[IDcs.index(IDc_single)]]
        except ValueError:
            comb = IDc_single.split(" ")
            combs = [[comb, IDc_single]]
        IDcs = [IDc_single]


for comb, IDc in combs:
    if IDc in ID_rename:
        IDcs[IDcs.index(IDc)] = ID_rename[IDc]
        IDc = ID_rename[IDc]
    print(IDc)
    comb += base_setting
    for i, key in enumerate(comb):
        if key in short_names:
            comb[i] = short_names[key]

    c = {}
    undefined = [key for key in comb if key not in keys]
    if len(undefined) > 0:
        raise ValueError("Undefined keys: {}".format(", ".join(undefined)))
    for k in keys:
        if k in comb:
            c[k] = True
        else:
            c[k] = False

    var_stag = xr.Dataset()
    #get staggered variables
    sec = ""
    if c["force_2nd_adv"]:
        fluxnames = ["R{}{}_TOT_MEAN_2ND".format(dim_dict[d], VAR) for d in ["x", "y"]]
        fluxnames.append("WW{}_TOT_MEAN_2ND".format(VAR))
        for fn, d in zip(fluxnames, xy):
            var_stag[d.upper()] = tools.stagger(dat_mean["{}_MEAN".format(VAR)], d, dat_mean[d + "_stag"], cyclic=cyclic[d])
        var_stag["Z"] = tools.stagger(dat_mean["{}_MEAN".format(VAR)], "bottom_top", grid["ZNW"], rename=True, **stagger_const)

        dim ="bottom_top"
        data = dat_mean["{}_MEAN".format(VAR)]
        stagger_const["CF1"]*data[{dim : 0}] + stagger_const["CF2"]*data[{dim : 1}] + stagger_const["CF3"]*data[{dim : 2}]
        sec = "_2ND"
    else:
        flux_names = None
        for d in [*XY, "Z"]:
            var_stag[d] = dat_mean["{}{}_MEAN".format(VAR, d)]

    flux_i, adv_i, vmean = tools.adv_tend(dat_mean, VAR, var_stag, grid, mapfac, cyclic, stagger_const,
                                   cartesian=c["cartesian"], recalc_w=c["recalc_w"], fluxnames=flux_names)

    adv_uncorrect = xr.Dataset()
    tend_i = total_tend
    if c["correct"] and c["cartesian"]:
        VARs = VAR + sec
        corr = dat_mean[["CORR_U{}".format(VARs), "CORR_V{}".format(VARs), "CORR_D{}DT{}".format(VAR, sec)]].to_array("dim")
        adv_i, tend_i = tools.cartesian_corrections(VAR, dim_stag, corr, var_stag, vmean, dat_mean["RHOD_MEAN"], dzdd, grid, adv_i, total_tend, cyclic, stagger_const)

    #add all forcings
    forcing_i = adv_i["Z"].sel(comp="tot", drop=True) + sources
    for d in XY:
        forcing_i = forcing_i + adv_i[d].sel(comp="tot", drop=True)

    rhodm_stag_m = dat_mean["RHOD_MEAN_STAG"]
    if hor_avg:
        flux_i = flux_i.mean(avg_dim)
        adv_i = adv_i.mean(avg_dim)
        forcing_i = forcing_i.mean(avg_dim)
        tend_i = tend_i.mean(avg_dim)
        rhodm_stag_m = dat_mean["RHOD_MEAN"].mean(avg_dim)

    #aggregate different IDs
    flux_id = flux_i.expand_dims(ID=[IDc])
    adv_id = adv_i.expand_dims(ID=[IDc])
    forcing_id = forcing_i.expand_dims(ID=[IDc])
    tend_id = tend_i.expand_dims(ID=[IDc])
    if flux is None:
        flux = flux_id
        adv = adv_id
        forcing = forcing_id
        tend = tend_id
    else:
        flux = xr.merge([flux, flux_id])
        adv = xr.merge([adv, adv_id])
        forcing = forcing.combine_first(forcing_id)
        tend = tend.combine_first(tend_id)

    #scatter plots
    fname = "{}_{}".format(IDi, IDc)
    if hor_avg:
        fname = fname + "_{}avg".format(avg_dim)

    fig = tools.scatter_tend_forcing(tend_i, forcing_i, var, rhodm_stag_m, plot_diff=plot_diff, hue=hue,
                               cut_boundaries=cut_boundaries, savefig=savefig, fname=fname)

adv = adv.sel(ID=IDcs)
flux = flux.sel(ID=IDcs)
forcing = forcing.sel(ID=IDcs)
tend = tend.sel(ID=IDcs)

adv_da = adv.to_array("dir")

flux_da = flux.sel(bottom_top_stag=flux.bottom_top_stag[:-1]).drop("bottom_top_stag").rename(bottom_top_stag="bottom_top").to_array("dir")
# zf = z.expand_dims(dir=flux_da.dir).copy()
# zf.loc["Z"] = zw[:,:-1].rename(bottom_top_stag="bottom_top").values
zm = grid["Z"]
if hor_avg:
    zm = grid["Z"].mean(avg_dim)
flux_da = flux_da.assign_coords(z=zm)

xl = len(dat_mean.x)
xvals = [int(xl/2), int(3*xl/4), xl]
xnames = None
if attrs["TOPO"] > 0:
    xnames = ["valley", "slope", "top"]


sys.exit()
#%% corrections
# sns.set_style("whitegrid")

# corr_tend = dcorr_dz.copy().drop("bottom_top").drop("zw")
# corr_tend["dim"] = ["X", "Y", "T"]
# corr_tend = corr_tend.loc[:, ["X", "Y", "T"]]
# corr_tend = corr_tend.sum("dim").expand_dims(part=["correction"])

# # adv_corr = adv_i.to_array("dim").loc[["X"],:].expand_dims(part=["corrected X-tendency"])
# adv_z = adv_i.to_array("dim").loc["Z",:].expand_dims(part=["native Z-tendency"]) + corr_tend[0].values
# adv_uncorr = adv_uncorrect.to_array("dim").loc["X",:].expand_dims(part=["native X-tendency"])
# # tend_t = tend_i.expand_dims({"dim" : ["T"]})
# # tend_t = xr.concat([adv_tot, tend_t], "dim")

# # dat = xr.concat([adv_corr, corr_tend], "part")
# dat = xr.concat([adv_z, corr_tend], "part")
# dat = xr.concat([adv_uncorr, dat], "part")/rhodm
# dat = dat.isel(x=xvals[1], Time=[-1]).sel(comp="tot")
# if xnames is not None:
#     dat["x"] = xnames[1]
# dat["Time"] = dat.Time.dt.time

# units = "10$^{-3}$" + units_dict[var] + "s$^{-1}$"
# dat = dat*1000
# var_n = "tendency ({})".format(units)

# df = dat.to_dataframe(name=var_n).reset_index()
# sns.set_style("whitegrid")

# grid = sns.relplot(data=df, x=var_n, y="z", col="Time", hue="part", kind="line", switch_axes=True,\
#                     facet_kws=dict(sharex="row", margin_titles=True, legend_out=False))

# grid.set_ylabels("height (m)")
# fname = "corr_{}".format(IDi)
# plt.savefig(figloc + "{}_budget/profiles/{}.pdf".format(var, fname),dpi=300, bbox_inches="tight")

#%% flux profiles

# flux_z = flux_i.Z.rename(bottom_top_stag="bottom_top", zw="z").drop("bottom_top")[:,:,:-1].expand_dims(part=["Z"])
# flux_x = flux_i.X.expand_dims(part=["X"])
# flux_z["z"] = flux_z.z.expand_dims(part=["Z"])
# flux_x["z"] = flux_x.z.expand_dims(part=["X"])
# zf = xr.concat([flux_x.z, flux_z.z], dim="part")
# flux = xr.concat([flux_x, flux_z], dim="part")/rhodm
# flux["z"] = zf
# flux = flux.isel(x=xvals[1], Time=[-1]).sel(comp="tot")
# flux["part"] = ["native X-flux", "native Z-flux"]

# units = units_dict[var] + "ms$^{-1}$"
# flux_n = "flux ({})".format(units)
# df = flux.to_dataframe(name=flux_n).reset_index()
# grid = sns.relplot(data=df, x=flux_n, y="z", hue="part", kind="line", switch_axes=True,\
#                    facet_kws=dict(sharex="row", margin_titles=True, legend_out=False))

# grid.set_ylabels("height (m)")
# fname = "flux_{}".format(IDi)
# plt.savefig(figloc + "{}_budget/profiles/{}.pdf".format(var, fname),dpi=300, bbox_inches="tight")

#%% flux cross sections
# sns.set_style("ticks")
# var = ["WDTH_TOT_MEAN", "WWTH_TOT_MEAN", "U_MEAN", "RUTH_TOT_MEAN"]
# labels = ["$\\langle \\rho w\\theta\\rangle$ (kg m$^{-2}$ Ks$^{-1})$",
#          "$\\langle  \\rho \\omega z_\eta \\theta\\rangle$ (kg m$^{-2}$ Ks$^{-1})$",
#          "$\\langle  u \\rangle$ (ms$^{-1})$",
#          "$\\langle  \\rho u\\theta\\rangle$ (kg m$^{-2}$ Ks$^{-1})$"]
# for v, l in zip(var, labels):
#     plt.figure()
#     if "zw" in dat_mean[v].coords:
#         y = "zw"
#     else:
#         y = "z"
#     if "W" in v:
#         vmin, vmax = -5, 5
#     else:
#         vmin, vmax = None, None
#     if "RU" in v:
#         dat = dat_mean[v]/dzdnw
#     else:
#         dat = dat_mean[v]
#     dat[-1].plot(y=y, vmin=vmin, size=3, aspect=2.1, vmax=vmax, cmap="RdBu_r", cbar_kwargs={"label": l})
#     plt.title("")
#     plt.xticks(np.arange(-5000,6000,2500))
#     plt.savefig("{}/th_budget/cross_sections/{}_{}.png".format(figloc, v, IDi), dpi=300, bbox_inches="tight")

# uth = tools.interpolate(dat_mean["RUTH_TOT_MEAN"]/dzdnw)
# uth[-1].plot(y="z")
# (-1000*(uth.diff("x")/dx))[-1,:,25].plot(y="z")
#%%plot profiles

kinds = ["adv"]
# kinds = ["flux", "adv", "adv_sum", "tend"]
# kinds = ["adv", "adv_sum", "tend"]
# kinds = ["tend"]
comps = ["tot","mean","res"]
comps = ["res"]#,"mean","res"]
IDs_all = list(forcing.ID.values)

img_type = "pdf"
close_fig = False
plot_diff = False #plot difference to reference
rel_diff = False #plot relative difference instead of absolute
marker = "."

zmax = None
times = adv.Time[1::2]
times = adv.Time[-1:]
divide_by_rho = True
#times = adv.Time[2:3]

for kind in kinds:
    IDs = IDs_all.copy()
    var_name = "tendency"
    units = units_dict[var] + "s$^{-1}$"
    if not divide_by_rho:
        units = "kg m$^{-3}$ "+ units
    if kind in ["adv", "adv_sum"]:
        dat = adv_da*1000
        var_name = "advective " + var_name
        units = "10$^{-3}$"+units
    elif kind == "flux":
        dat = flux_da
        var_name = "${}$ flux".format(tex_names[var])
        if divide_by_rho:
            units = "{} m s$^{-1}$"
        else:
            units = "kg m$^{-2}$ %ss$^{-1}$" % units_dict[var]

    elif kind == "tend":
        dat = 1000*forcing.expand_dims(comp=["forcing"]).reindex(comp=["forcing", "tend"])
        dat.loc["tend"] = tend*1000
        comps = ["forcing"]
        var_name = "total " + var_name
        units = "10$^{-3}$"+units

    else:
        raise ValueError("Variable {} not understood!".format(kind))


    if divide_by_rho:
        dat = dat/dat_mean["RHOD_MEAN"]
    if kind == "tend":
        dirs = [""]
    elif kind == "adv_sum":
        dat = dat.sum("dir")
        dirs = [""]
    else:
        dirs = dat.dir.values
        if "correct 2nd" in IDs:
            IDs.remove("correct 2nd")

    if kind != "tend":
        IDs.remove('correct recalc_w')
    # dat = dat.sel(ID=IDs)

    zb = grid["Z"].reindex(x=[*grid["Z"].x,-grid["Z"].x[0]])
    zb = zb.where(zb.x < zb.x[-1], zb.isel(x=0))
    dat = dat.reindex(x=zb.x)
    dat = dat.where(dat.x < dat.x[-1], dat.isel(x=0))
    dat["z"] = zb

    dat = dat.sel(Time=times).isel(x=xvals, drop=True)
    dat = dat.sel(bottom_top=dat.bottom_top[:-1])
    if zmax is not None:
        dat = dat.where(dat.z < zmax).dropna("bottom_top", "all")
    if not hor_avg:
        dat = dat.sel({avg_dim: 0}, drop=True)
    #    dat = dat.sel(dir=["Z"])#, comp=["res"])
    if xnames is not None:
        dat["x"] = xnames
    ylim = dat.z.max().values

    for comp in comps:
        for time in times.values:
            for i,d in enumerate(dirs):
                fig,axes =plt.subplots(figsize=(12,6),nrows=len(dat.x), ncols=len(IDs), sharex="col", sharey=True)
                for j,x in enumerate(dat.x):
                    for k,ID in enumerate(IDs):
                        ax = axes[j,k]
                        datp = dat
                        if "dir" in dat.dims:
                            datp = dat.sel(dir=d)
                        if kind == "tend":
                            ref =  dat.sel(comp="tend",Time=time,x=x,ID=ID)
                        else:
                            ref =  datp.sel(ID='correct recalc_w', comp=comp,Time=time,x=x)

                        datp_i = datp.sel(comp=comp,Time=time,x=x,ID=ID)
                        label = "simplified"
                        if plot_diff:
                            datp_i = datp_i - ref
                            label += " - reference"
                            if rel_diff:
                                datp_i = datp_i/ref
                                label = "(" + label + ")/reference"
                        grid = datp_i.plot(y="z", ax=ax, label=label, linewidth=2, marker=marker)
                        if not plot_diff:
                            if kind == "tend":
                                ref.plot(y="z", ax=ax,linewidth=1, label="total tendency", alpha=1)
                            else:
                                ref.plot(y="z", ax=ax, label="reference ({})".format(dat.ID[-1].values), linewidth=1, alpha=1)

                        if x == dat.x[0]:
                            ax.set_title(ID)
                        else:
                            ax.set_title("")

                        if x == dat.x[-1]:
                            ax.set_xlabel("{} ({})".format(var_name, units))
                        ax.vlines(0,-500,ylim, linestyles="solid", linewidth=1, color="grey")
                        dati = dat.sel(ID=ID,comp=comp)
                        if plot_diff:
                            if kind == "tend":
                                ref =  dat.sel(comp="tend",ID=ID)
                            else:
                                ref =  datp.sel(ID='correct recalc_w', comp=comp)
                            dati = dati - ref
                            if rel_diff:
                                dati = dati/ref
                        dmin, dmax = dati.min(), dati.max()
                        r = dmax - dmin
                        ax.set_xlim(dmin - 0.02*r, dmax + 0.02*r)
                    ax.text(1.05, 0.5, 'x={}'.format(x.values), rotation="vertical",
                            verticalalignment="center", transform=ax.transAxes)

                    ax.set_ylim(-500, ylim)
                plt.legend(loc="upper left", bbox_transform=fig.transFigure, bbox_to_anchor=(0.1, 1.02))
                title = kind
                fname = "{}_{}".format(IDi, kind)
                if kind != "tend":
                    fname += "_" + comp

                if d != "":
                    title +=  "  dir = {}".format(d)
                    fname += "_" + d
                fig.suptitle(title)
                plt.show()
                fig.suptitle("")

                if hor_avg:
                    fname += "_{}avg".format(avg_dim)
                fname += "_{0}:{1:02}".format(dat.sel(Time=time).Time.dt.hour.values, dat.sel(Time=time).Time.dt.minute.values)
                fig.savefig(figloc + "{}_budget/profiles/{}.{}".format(var, fname, img_type),dpi=300, bbox_inches="tight")
                if close_fig:
                    plt.close()
                # fluxl.isel(ID=[i,-1]).plot(y="z", col="x", row="dir", hue="ID")

sys.exit()

#%%profiles for all tendency components
xl = len(adv_da.x)
rolling_xwindow = 90

ID = 'correct recalc_w'
# ID = adv_da.ID.values
comps = ["adv_r", "mean", "trb", "net", "forcing"]
comps = ["MEAN+RES", "MEAN", "RES", "NET", "SGS", "radiation", "forcing"]
comps = ["MEAN+RES", "MEAN", "RES", "NET", "SGS", "radiation"]
comps = ["MEAN", "RES", "NET", "SGS", "radiation"]
# comps = ["MEAN+RES", "MEAN", "RES", "NET", "SGS"]
# comps = ["forcing"]
# comps = None
#xvals = [0, int(xl/4), int(xl/2)]
#xnames = ["top", "slope", "valley"]

ref = xr.concat([adv_da.sel(ID=ID), sgs.expand_dims(comp=["sgs"])],dim="comp").drop("zw")
# ref["comp"] = ["adv_r", "mean", "trb", "trb_s"]
ref["comp"] = ["MEAN+RES", "MEAN", "RES", "SGS"]
ref_sum = ref.sum("dir")

phys = []
if var == "th":
    if attrs["MP_PHYSICS"] > 0:
        phys.append(dat_mean["T_TEND_MP_MEAN"].expand_dims(comp=["microphysics"]))
    if (attrs["RA_LW_PHYSICS"] > 0) or (attrs["RA_SW_PHYSICS"] > 0):
        phys.append((dat_mean["T_TEND_RADSW_MEAN"] + dat_mean["T_TEND_RADLW_MEAN"]).expand_dims(comp=["radiation"]))
else:
    phys.append(dat_mean["Q_TEND_MP_MEAN"].expand_dims(comp=["microphysics"]))

qfac = 1
if (var == "th") and attrs["USE_THETA_M"] and (not attrs["OUTPUT_DRY_THETA_FLUXES"]):
    qfac = 1 + metconst.Rv.m/metconst.Rd.m*dat_mean["Q_MEAN"]
phys = xr.concat(phys, dim="comp")*dat_mean["RHOD_MEAN"]*qfac

ref_tot = xr.concat([ref_sum, phys, forcing.drop("z").sel(ID=ID).expand_dims(comp=["forcing"]), tend.sel(ID=ID).expand_dims(comp=["NET"])],dim="comp", coords="minimal")
ref_tot = ref_tot/dat_mean["RHOD_MEAN"]

zb = grid["Z"].reindex(x=[*grid["Z"].x,-grid["Z"].x[0]])
zb = zb.where(zb.x < zb.x[-1], zb.isel(x=0))

ref_tot = ref_tot.reindex(x=zb.x)
ref_tot = ref_tot.where(ref_tot.x < ref_tot.x[-1], ref_tot.isel(x=0))
ref_tot["z"] = zb
ref_tot["hgt"] = ref_tot["z"] - ref_tot["z"].isel(bottom_top=0)


if rolling_xwindow is not None:
    ref_tot = ref_tot.rolling(x=rolling_xwindow).mean()


dat = ref_tot.where(ref_tot.hgt < 2000).dropna("bottom_top", "all")
dat = 1000*dat.isel(Time=-1, x=xvals, drop=True).drop("ID")
if xnames is not None:
    dat["x"] = xnames
if comps is not None:
    dat = dat.sel(comp=comps)

# cmap = plt.get_cmap("tab10")
colors = {"MEAN+RES":"gray", "MEAN":"tab:blue", "RES":"tab:red",
          "SGS":"tab:orange", "forcing":"black", "NET":"violet",
          "radiation":"tab:green"}
# c = tuple((mpl.colors.ColorConverter.to_rgb(color) for color in colors))

datp = dat.copy()
datp.loc[{"comp":"NET"}] = np.nan
df = datp.to_dataframe(name="tend").reset_index()
# df["dashed"] = np.in1d(df["comp"], ["adv_r", "trb_s"])
sns.set_style("whitegrid")

row = None#"ID"
hue = "comp"
col = "x"
palette = colors
sharex = False

##%%profiles for all tendency components
# row = "comp"
# hue = "x"
# col = "ID"
# palette = None
# sharex = False

grid = sns.relplot(data=df, aspect=0.6, height=3.5, kind="line", x="tend", y="hgt",
                   row=row, col=col, hue=hue, facet_kws={"sharex":sharex, "margin_titles":True}, palette=palette, switch_axes=True)



grid.set_xlabels("$%s$ tendency (10$^{-3}$ %ss$^{-1}$) components" % (tex_names[var], units_dict[var]))
grid.set_ylabels("height above ground (m)")

for i,ax in enumerate(grid.axes.flatten()):
    ax2 = ax.twiny()
    ax.set_title("")
    net =  dat.sel(comp="NET", drop=True)
    net.isel(x=i).plot.line(ax=ax2, y="hgt", c=colors["NET"])
    ax2.set_xlim(net.min(), net.max())
    if i != 1:
        ax.set_xlabel("")
        ax2.set_xlabel(" ")
    else:
        ax2.set_xlabel("NET $%s$ tendency (10$^{-3}$ %ss$^{-1}$)" % (tex_names[var], units_dict[var]))
plt.savefig(figloc + "{}_budget/profiles/all_comps_{}_{}.pdf".format(var, IDi, ID), bbox_inches="tight")
# dat.plot.line(x="hgt", col="x", hue="comp")
#dat.plot.line(y="hgt", col="x", hue="comp")
#dat[-1].plot(hue="x", y="hgt")
#%%cross sections
for zcoord, v in zip(["z", "zw"],["U", "W"]):
    plt.figure()
    dat_mean["{}_MEAN".format(v)][-1].plot(y=zcoord)
    p = (dat_mean["TH_MEAN"]+300)[-1].plot.contour(y="z", colors="k", levels=np.arange(298, 318, 0.5), linewidths=0.2, add_labels=True)
    p.clabel(levels=np.arange(300, 318, 5), fmt='%1.0f')
    plt.savefig(figloc + "cross_sections/{}_{}.png".format(v, IDi), dpi=300)

# p = (dat_mean["TH_MEAN"] - dat_mean["TH_MEAN"].sel(x=0).)[1].plot.contour(y="z", colors="k", levels=np.arange(298, 318), linewidths=0.2, add_labels=True)

#%% error plot
diff = (((tend - forcing)**2)[0].mean(["Time","y","x_stag"]))**0.5
# diff.plot(label="without acoustic step tendency")
diff.plot(label="with acoustic step tendency")
plt.title("")
plt.legend()
plt.ylabel("RMSE (kg m$^{-2}$ K s$-1}$")
plt.xlabel("$\\eta$")
# plt.savefig("{}/th_budget/profiles/rmse_with(out)smallstep.pdf".format(figloc))


#%%w vs wd
# fig, ax = plt.subplots()
# pdatf = dat_mean["WD_MEAN"].stack(s=dat_mean["WD_MEAN"].dims)
# color = -pdatf.bottom_top_stag
# p = plt.scatter(dat_mean["WD_MEAN"], dat_mean["W_MEAN"], s=5, c=color, cmap="cool")
# plt.ylabel("$w $ (ms$^{-1}$)")
# plt.xlabel("$\\hat{w} $ (ms$^{-1}$)")

# minmax = [dat_mean["WD_MEAN"].min().values, dat_mean["WD_MEAN"].max().values]
# dist = minmax[1] - minmax[0]
# minmax[0] -= 0.03*dist
# minmax[1] += 0.03*dist
# plt.plot(minmax, minmax, c="k")

# cax = fig.add_axes([0.84,0.1,0.1,0.8], frameon=False)
# cax.set_yticks([])
# cax.set_xticks([])
# cb = plt.colorbar(p,ax=cax,label="$\eta$")
# cb.set_ticks(np.arange(-0.8,-0.2,0.2))
# cb.set_ticklabels(np.linspace(0.8,0.2,4).round(1))
# ax.set_xlim(minmax)
# ax.set_ylim(minmax)
# plt.savefig(figloc + "{}_budget/scatter/w_wd_{}.png".format(var, IDi),dpi=300, bbox_inches="tight")
# plt.scatter(dat_mean["WWTH_TOT_MEAN"], dat_mean["CORR_UTH"])
