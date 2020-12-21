#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 09:35:43 2020

@author: Matthias Göbel
"""
from wrflux import tools, plotting
import xarray as xr

def test_budget(datout, thresh=0.01, plot=True, **plot_kws):
    failed = False
    for ID in ["native", "cartesian correct"]:
        ref = datout["tend"].sel(comp="tendency", ID=ID, drop=True)
        dat = datout["tend"].sel(comp="forcing", ID=ID, drop=True)
        e = tools.max_error_scaled(dat, ref)
        if e > thresh:
            log = " tendency vs forcing for ID='{}': maximum error in forcing more than {}%: {:.3f}%".format(ID, thresh*100, e*100)
            print(log)
            if plot:
                # plotting.scatter_tend_forcing(dat, ref, var, **plot_kws)
                plotting.scatter_hue(dat, ref, title=log, **plot_kws)
            failed = True
    return failed


def test_2nd(datout, thresh=0.01, thresh_sum=0.02, plot=True, **plot_kws):
    base = "cartesian"
    failed = False
    for correct in [False, True]:
        for sum_dir in [False, True]:
            ID = base
            if correct:
                ID += " correct"
            ID2 = ID + " 2nd"
            for i in [ID, ID2]:
                if i not in datout["adv"].ID:
                    raise ValueError("Could not find output of budget method '{}'".format(i))

            data = datout["adv"]
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

def test_decomp_sum(datout, thresh=0.03, plot=True, **plot_kws):
#native sum vs. cartesian sum
    ID = "native"
    ID2 = "cartesian correct"
    data = datout["adv"].sel(dir="sum")
    ref = data.sel(ID=ID)
    dat = data.sel(ID=ID2)
    e = tools.max_error_scaled(dat, ref)
    failed = False
    if e > thresh:
        log = " '{}' vs '{}': maximum error in adv sum more than {}%: {:.3f}%".format(ID, ID2, thresh*100, e*100)
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
            dnan = tools.find_nans(d[dv])
            if sum(dnan.shape) != 0:
                print("\nWARNING: found NaNs in {} :\n{}".format(v, dnan.coords))
                failed = True

    return failed
#%%recalc_w vs not recalc_w
#TODOm delete
    # base_i = base + " correct"
    # ID = base_i + " recalc_w"
    # ID2 = base_i
    # for sum_dir in [False, True]:
    #     data = datout["adv"]
    #     if sum_dir:
    #         data = data.sel(dir="sum")
    #         thresh = .2 #TODO really need such high threshold? problem with mean?
    #         sum_s = "sum"
    #     else:
    #         thresh = .5
    #         sum_s = ""
    #     ref = data.sel(ID=ID)
    #     dat = data.sel(ID=ID2)
    #     # dat = dat_mean["FQZ_ADV_MEAN"]
    #     # ref = dat_mean["FQZ_ADV_MEAN_PROG"]

    #     e = tools.max_error_scaled(dat, ref)*100
    #     plot_kws = dict(
    #         s=20,
    #         # discrete=True,
    #         # plot_diff=True,
    #         # iloc= {"x" : slice(1,5)},
    #         # iloc= {"bottom_top" : slice(0,5)},
    #         hue="comp",
    #          # plot_iloc =
    #     )
    #     if e > thresh:
    #         log = " '{}' vs '{}': maximum error in adv {} more than {}%: {:.3f}%".format(ID, ID2, sum_s, thresh, e)
    #         print(log)
    #         plotting.scatter_hue(dat, ref, title=log, **plot_kws)


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
