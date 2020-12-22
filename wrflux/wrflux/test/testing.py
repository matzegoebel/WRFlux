#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 09:35:43 2020

@author: Matthias Göbel
"""
from wrflux import tools, plotting
import xarray as xr

def test_budget(tend, thresh=0.01, plot=True, **plot_kws):
    failed = False
    for ID in ["native", "cartesian correct"]:
        ref = tend.sel(comp="tendency", ID=ID, drop=True)
        dat = tend.sel(comp="forcing", ID=ID, drop=True)
        e = tools.max_error_scaled(dat, ref)
        if e > thresh:
            log = " tendency vs forcing for ID='{}': maximum error in forcing more than {}%: {:.3f}%".format(ID, thresh*100, e*100)
            print(log)
            if plot:
                # plotting.scatter_tend_forcing(dat, ref, var, **plot_kws)
                plotting.scatter_hue(dat, ref, title=log, **plot_kws)
            failed = True
    return failed


def test_2nd(adv, thresh=0.01, thresh_sum=0.02, plot=True, **plot_kws):
    base = "cartesian"
    failed = False
    for correct in [False, True]:
        for sum_dir in [False, True]:
            ID = base
            if correct:
                ID += " correct"
            ID2 = ID + " 2nd"
            for i in [ID, ID2]:
                if i not in adv.ID:
                    raise ValueError("Could not find output of budget method '{}'".format(i))

            data = adv
            sum_s = ""
            t = thresh
            if sum_dir:
                data = data.sel(dir="sum")
                sum_s = "sum"
                t = thresh_sum
            ref = data.sel(ID=ID)
            dat = data.sel(ID=ID2)
            e = tools.max_error_scaled(dat, ref)
            if e > t:
                log = "'{}' vs '{}': maximum error in adv {} more than {}%: {:.3f}%".format(ID, ID2, sum_s, thresh*100, e*100)
                print(log)
                if plot:
                    #TODO: dat.name = ...
                    plotting.scatter_hue(dat, ref, title=log, **plot_kws)
                failed = True
    return failed

def test_decomp_sumdir(adv, corr, thresh=0.01, plot=True, **plot_kws):
#native sum vs. cartesian sum
    ID = "native"
    ID2 = "cartesian correct"
    data = adv.sel(dir="sum")
    ref = data.sel(ID=ID)
    dat = data.sel(ID=ID2) + corr.sel(ID=ID2, dir="T")
    e = tools.max_error_scaled(dat, ref)
    failed = False
    if e > thresh:
        log = " '{}' vs '{}': maximum error in adv sum more than {}%: {:.3f}%".format(ID, ID2, thresh*100, e*100)
        print(log)
        if plot:
            plotting.scatter_hue(dat, ref, title=log, **plot_kws)
        failed = True
    return failed

def test_dz_out(adv, plot=True, **plot_kws):
    failed = False
    ID = "cartesian correct"
    if "bottom_top" in adv.dims:
        eta = adv["bottom_top"][1].values
    else:
        eta = adv["bottom_top_stag"][2].values

    for corr_varz, threshs in zip([True, False],[[0.99, 0.99, 0.9999, 0.99], [0.92, 0.99, 0.9999, 0.99]]) :
        ID2 = "cartesian correct dz_out"
        if corr_varz:
            ID2 = ID2 + " corr_varz"
        for thresh, loc in zip(threshs,#TODOm
                       [{"dir" : ["X", "sum"]}, {"comp" : "mean"}, {"dir" : ["Y", "Z"]}, {"bottom_top" : slice(eta, None)}]):
            loc = tools.correct_dims_stag(loc, adv)
            ref = adv.sel(ID=ID, **loc)
            dat = adv.sel(ID=ID2, **loc)
            e = tools.nse(dat, ref).values
            if e < thresh:
                log = "'{}' vs '{}' for loc={}: NSE in adv less than {}%: {:.3f}%".format(ID, ID2, loc, thresh*100, e*100)
                print(log)
                if plot:
                    plotting.scatter_hue(dat, ref, title=log, **plot_kws)
                failed = True
    return failed

def test_decomp_sumcomp(adv, thresh=0.01, plot=True, **plot_kws):
#native sum vs. cartesian sum
    failed = False
    for ID in adv.ID:
        ref = adv.sel(ID=ID, comp="adv_r")
        dat = adv.sel(ID=ID, comp=["mean", "trb_r"]).sum("comp")
        e = tools.max_error_scaled(dat, ref)
        if e > thresh:
            log = " 'mean + trb_r' vs 'adv_r': maximum error in adv sum more than {}%: {:.3f}%".format(thresh*100, e*100)
            print(log)
            if plot:
                plotting.scatter_hue(dat, ref, title=log, **plot_kws)
            failed = True
    return failed


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


def test_w(dat_inst, thresh=0.01, plot=True, **plot_kws):
    #%% w vs wd
    dat = dat_inst["W"]
    ref = dat_inst["W_DIAG"]
    # dat = dat_mean["W_MEAN"]
    # ref = dat_mean["WD_MEAN"]
    e = tools.max_error_scaled(dat, ref)
    failed = False
    if e > thresh:
        log = " w vs wd: maximum difference more than {}%: {:.3f}%".format(thresh*100, e*100)
        print(log)
        if plot:
            plotting.scatter_hue(dat, ref, title=log, **plot_kws)
        failed = True
    return failed


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
