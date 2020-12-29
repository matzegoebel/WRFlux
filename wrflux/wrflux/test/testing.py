#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 09:35:43 2020

@author: Matthias Göbel
"""
from wrflux import tools, plotting
import xarray as xr
xr.set_options(arithmetic_join="exact")
xr.set_options(keep_attrs=True)

def test_budget(tend, forcing, avg_dims_error=None, thresh=0.9995, loc=None, iloc=None, plot=True, **plot_kws):

    failed = False
    err = []
    for ID in ["native", "cartesian correct"]:

        ref = tend.sel(ID=ID, drop=True)
        dat = forcing.sel(ID=ID, drop=True)
        dat = tools.loc_data(dat, loc=loc, iloc=iloc)
        ref = tools.loc_data(ref, loc=loc, iloc=iloc)
        e = tools.nse(dat, ref, dim=avg_dims_error).min().values
        err.append(e)
        if e < thresh:
            log = " tendency vs forcing for ID='{}': NSE less than {}%: {:.3f}%".format(ID, thresh*100, e*100)
            print(log)
            if plot:
                # plotting.scatter_tend_forcing(dat, ref, var, **plot_kws)
                plotting.scatter_hue(dat, ref, title=log, **plot_kws)
            failed = True
    return failed, min(err)


def test_2nd(adv, avg_dims_error=None, thresh=0.998, loc=None, iloc=None, plot=True, **plot_kws):

    base = "cartesian"
    failed = False
    err = []
    for correct in [False, True]:
        ID = base
        if correct:
            ID += " correct"
        ID2 = ID + " 2nd"
        for i in [ID, ID2]:
            if i not in adv.ID:
                raise ValueError("Could not find output of budget method '{}'".format(i))

        data = adv

        ref = data.sel(ID=ID)
        dat = data.sel(ID=ID2)
        dat = tools.loc_data(dat, loc=loc, iloc=iloc)
        ref = tools.loc_data(ref, loc=loc, iloc=iloc)
        e = tools.nse(dat, ref, dim=avg_dims_error).min().values
        err.append(e)
        if e < thresh:
            log = "'{}' vs '{}': NSE less than {}%: {:.3f}%".format(ID, ID2, thresh*100, e*100)
            print(log)
            if plot:
                #TODO: dat.name = ...
                plotting.scatter_hue(dat, ref, title=log, **plot_kws)
            failed = True
    return failed, min(err)

def test_decomp_sumdir(adv, corr, avg_dims_error=None, thresh=0.995, loc=None, iloc=None, plot=True, **plot_kws):
#native sum vs. cartesian sum

    ID = "native"
    ID2 = "cartesian correct"
    data = adv.sel(dir="sum")
    ref = data.sel(ID=ID)
    dat = data.sel(ID=ID2) + corr.sel(ID=ID2, dir="T")
    dat = tools.loc_data(dat, loc=loc, iloc=iloc)
    ref = tools.loc_data(ref, loc=loc, iloc=iloc)
    e = tools.nse(dat, ref, dim=avg_dims_error).min().values
    failed = False
    if e < thresh:
        log = " '{}' vs '{}': NSE less than {}%: {:.3f}%".format(ID, ID2, thresh*100, e*100)
        print(log)
        if plot:
            plotting.scatter_hue(dat, ref, title=log, **plot_kws)
        failed = True
    return failed, e

def test_dz_out(adv, avg_dims_error=None, thresh=0.95, loc=None, iloc=None, plot=True, **plot_kws):
    failed = False
    ID = "cartesian correct"
    ID2 = "cartesian correct dz_out corr_varz"

    ref = adv.sel(ID=ID)
    dat = adv.sel(ID=ID2)
    dat = tools.loc_data(dat, loc=loc, iloc=iloc)
    ref = tools.loc_data(ref, loc=loc, iloc=iloc)
    e = tools.nse(dat, ref, dim=avg_dims_error).min().values
    if e < thresh:
        log = "'{}' vs '{}': NSE in adv less than {}%: {:.3f}%".format(ID, ID2, thresh*100, e*100)
        print(log)
        if plot:
            plotting.scatter_hue(dat, ref, title=log, **plot_kws)
        failed = True
    return failed, e

def test_decomp_sumcomp(adv, avg_dims_error=None, thresh=0.9999999999, loc=None, iloc=None, plot=True, **plot_kws):
#native sum vs. cartesian sum
    failed = False
    err = []
    for ID in adv.ID.values:
        ref = adv.sel(ID=ID, comp="adv_r")
        dat = adv.sel(ID=ID, comp=["mean", "trb_r"]).sum("comp")
        dat = tools.loc_data(dat, loc=loc, iloc=iloc)
        ref = tools.loc_data(ref, loc=loc, iloc=iloc)
        e = tools.nse(dat, ref, dim=avg_dims_error).min().values
        err.append(e)
        if e < thresh:
            log = " 'mean + trb_r' vs 'adv_r': NSE less than {}%: {:.3f}%".format(thresh*100, e*100)
            print(log)
            if plot:
                plotting.scatter_hue(dat, ref, title=log, **plot_kws)
            failed = True
    return failed, min(err)


def test_nan(datout):

    failed = False
    for f,d in datout.items():
        if f == "grid":
            continue
        da = False
        if type(d) == xr.core.dataarray.DataArray:
            d = d.to_dataset(name=f)
            da = True
        for dv in d.data_vars:
            if da:
                v = f
            else:
                v = "{}/{}".format(f, dv)
            dnan = tools.find_bad(d[dv])
            if sum(dnan.shape) != 0:
                print("\nWARNING: found NaNs in {} :\n{}".format(v, dnan.coords))
                failed = True

    return failed


#%% mean vs adv_r
    # for ID in adv.ID:
    #     dat = adv.sel(ID=ID, comp="mean")
    #     ref = adv.sel(ID=ID, comp="adv_r")
    #     e = tools.max_error_scaled(dat, ref)*100 #TODO use scaled rmse?
    #     thresh = 15
    #     if e > thresh:
    #         log = " mean vs adv_r: maximum difference more than {}%: {:.3f}% ; for ID: {}".format(thresh, e, ID.values)
    #         print(log)
    #         plotting.scatter_hue(dat, ref, hue="dir", title=log)
    # adv_sum = adv.sum("dir")
    # adv_sum[:,:,:,-1,[0,2,5]].plot(hue="comp",row="ID", y="bottom_top", col="x")
    # adv[-1,:,:,:,-1,[0,2,5]].plot(hue="comp", row="dir", y="bottom_top", col="x")
    #TODO: why is turbulent so strong on slope?


def test_w(dat_inst, avg_dims_error=None, thresh=0.995, loc=None, iloc=None, plot=True, **plot_kws):
    dat_inst = tools.loc_data(dat_inst, loc=loc, iloc=iloc)

    dat = dat_inst["W"]
    ref = dat_inst["W_DIAG"]
    # dat = dat_mean["W_MEAN"]
    # ref = dat_mean["WD_MEAN"]
    e = tools.nse(dat, ref, dim=avg_dims_error).min().values
    failed = False
    if e < thresh:
        log = " w vs wd: NSE less than {}%: {:.3f}%".format(thresh*100, e*100)
        print(log)
        if plot:
            plotting.scatter_hue(dat, ref, title=log, **plot_kws)
        failed = True
    return failed, e


def check_bounds(dat_mean, attrs, var):
    for dim in ["x", "y"]:
        if not attrs["PERIODIC_{}".format(dim.upper())]:
            for comp in ["ADV", "SGS"]:
                for flx_dir in ["X", "Y", "Z"]:
                    flx_name = "F{}{}_{}_MEAN".format(var.upper(), flx_dir, comp)
                    flx = dat_mean[flx_name]
                    if (comp == "SGS") and (flx_dir == "Z"):
                        #sgs surface flux is filled everywhere
                        flx = flx[:,1:]
                    dims = dim
                    if dim not in flx.dims:
                        dims = dim + "_stag"
                    if not (flx[{dims : [0,-1]}] == 0).all():
                        print("For non-periodic BC in {0} direction, {1} should be zero on {0} boundaries!".format(dim, flx_name))
