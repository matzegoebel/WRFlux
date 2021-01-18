#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 09:35:43 2020

@author: Matthias Göbel
"""
from wrflux import tools, plotting


def test_budget(tend, forcing, avg_dims_error=None, thresh=0.9993,
                loc=None, iloc=None, plot=True, **plot_kws):

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
            log = "test_budget for ID='{}': min. NSE less than {}: {:.5f}\n".format(ID, thresh, e)
            print(log)
            if plot:
                dat = dat.assign_attrs(description=dat.description[:2] + "forcing")
                plotting.scatter_hue(dat, ref, title=log, **plot_kws)
            failed = True
    return failed, min(err)


def test_decomp_sumdir(adv, corr, avg_dims_error=None, thresh=0.999999,
                       loc=None, iloc=None, plot=True, **plot_kws):
    # native sum vs. cartesian sum

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
        log = "test_decomp_sumdir, {}: min. NSE less than {}: {:.7f}".format(dat.description, thresh, e)
        print(log)
        if plot:
            dat = dat.assign_attrs(description="cartesian")
            ref = ref.assign_attrs(description="native")
            plotting.scatter_hue(dat, ref, title=log, **plot_kws)
        failed = True
    return failed, e


def test_decomp_sumcomp(adv, avg_dims_error=None, thresh=0.9999999999,
                        loc=None, iloc=None, plot=True, **plot_kws):
    # native sum vs. cartesian sum
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
            log = "decomp_sumcomp, {} (XYZ) for ID={}: min. NSE less than {}: {:.11f}".format(
                dat.description, ID, thresh, e)
            print(log)
            if plot:
                ref = ref.assign_attrs(description="adv_r")
                dat = dat.assign_attrs(description="mean + trb_r")
                plotting.scatter_hue(dat, ref, title=log, **plot_kws)
            failed = True
    return failed, min(err)


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
        log = "test_dz_out, {} (XYZ): min. NSE less than {}: {:.5f}".format(dat.description, thresh, e)
        print(log)
        if plot:
            dat = dat.assign_attrs(description="dz_out corr_varz")
            ref = ref.assign_attrs(description="reference corr.")
            plotting.scatter_hue(dat, ref, title=log, **plot_kws)
        failed = True
    return failed, e


def test_2nd(adv, avg_dims_error=None, thresh=0.999, loc=None, iloc=None, plot=True, **plot_kws):

    base = "cartesian"
    failed = False
    err = []
    for correct in [False, True]:
        ID = base
        without = "out"
        if correct:
            without = ""
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
            log = "test_2nd with{} corrections, {} (XYZ): min. NSE less than {}: {:.5f}".format(
                without, dat.description, thresh, e)
            print(log)
            if plot:
                ref = ref.assign_attrs(description="correct order")
                dat = dat.assign_attrs(description="2nd order")
                plotting.scatter_hue(dat, ref, title=log, **plot_kws)
            failed = True
    return failed, min(err)


def test_w(dat_inst, avg_dims_error=None, thresh=0.999, loc=None, iloc=None, plot=True, **plot_kws):
    dat_inst = tools.loc_data(dat_inst, loc=loc, iloc=iloc)

    ref = dat_inst["W"]
    dat = dat_inst["W_DIAG"]
    e = tools.nse(dat, ref, dim=avg_dims_error).min().values
    failed = False
    if e < thresh:
        log = "test_w: min. NSE less than {}: {:.5f}".format(thresh, e)
        print(log)
        if plot:
            plotting.scatter_hue(dat, ref, title=log, **plot_kws)
        failed = True
    return failed, e


def test_nan(datout, cut_bounds=None):
    failed = False
    for f, d in datout.items():
        if f == "grid":
            continue
        da = False
        if type(d) == tools.xr.core.dataarray.DataArray:
            d = d.to_dataset(name=f)
            da = True
        if cut_bounds is not None:
            for dim in cut_bounds:
                for stag in ["", "_stag"]:
                    dim = dim + stag
                    if dim in d.dims:
                        d = d[{dim: slice(1, -1)}]
        for dv in d.data_vars:
            if da:
                v = f
            else:
                v = "{}/{}".format(f, dv)
            dnan = tools.find_bad(d[dv])
            if (dnan is not None) and (sum(dnan.shape) != 0):
                print("\nWARNING: found NaNs in {} :\n{}".format(v, dnan.coords))
                failed = True

    return failed


def test_y0(adv):
    failed = False
    dims = [d for d in adv.dims if d not in ["dir", "ID"]]
    f = abs((adv.sel(dir="Y") / adv.sel(dir="X"))).mean(dims)
    for ID, thresh in zip(["native", "cartesian correct"], [1e-6, 5e-2]):
        fi = f.loc[ID].values
        if fi > thresh:
            print("test_y0 failed for ID={}!: mean(|adv_y/adv_x|) = {} > {}".format(ID, fi, thresh))
            failed = True
    return failed

# TODO: reimplement?
# def check_bounds(dat_mean, attrs, var):
#     for dim in ["x", "y"]:
#         if not attrs["PERIODIC_{}".format(dim.upper())]:
#             for comp in ["ADV", "SGS"]:
#                 for flx_dir in ["X", "Y", "Z"]:
#                     flx_name = "F{}{}_{}_MEAN".format(var.upper(), flx_dir, comp)
#                     flx = dat_mean[flx_name]
#                     if (comp == "SGS") and (flx_dir == "Z"):
#                         #sgs surface flux is filled everywhere
#                         flx = flx[:,1:]
#                     dims = dim
#                     if dim not in flx.dims:
#                         dims = dim + "_stag"
#                     if not (flx[{dims : [0,-1]}] == 0).all():
#                         print("For non-periodic BC in {0} direction, {1} should be zero on {0} boundaries!".format(dim, flx_name))
