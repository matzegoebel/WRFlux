#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for automatic testing of WRFlux output.

@author: Matthias Göbel
"""
from wrflux import tools, plotting
import pandas as pd
import xarray as xr
import os
from pathlib import Path

all_tests = ["budget", "decomp_sumdir", "decomp_sumcomp",
             "dz_out", "adv_2nd", "w", "Y=0", "NaN", "dim_coords",
             "no_model_change"]
# %% test functions

def test_budget(tend, forcing, avg_dims_error=None, thresh=0.9993,
                loc=None, iloc=None, plot=True, **plot_kws):
    """
    Test closure of budget: tend = forcing.

    Only the budget methods "native" and "cartesian" are tested.
    The test fails if the Nash-Sutcliffe efficiency coefficient (NSE)
    is below the given threshold. If avg_dims_error is given, the averaging in the
    NSE calculation is only carried out over these dimensions. Afterwards the minimum NSE
    value is taken over the remaining dimensions.

    Parameters
    ----------
    tend : xarray DataArray
        Total tendency.
    forcing : xarray DataArray
        Total forcing.
    avg_dims_error : str or list of str, optional
        Dimensions over which to calculate the NSE. The default is None.
    thresh : float, optional
        Threshold value for NSE below which the test fails
    loc : dict, optional
        Mapping for label based indexing before running the test. The default is None.
    iloc : dict, optional
        Mapping for integer-location based indexing before running the test. The default is None.
    plot : bool, optional
        Create scatter plot if test fails. The default is True.
    **plot_kws :
        keyword arguments passed to plotting.scatter_hue.

    Returns
    -------
    failed : bool
        Test failed.
    err : float
        Test statistic NSE

    """
    failed = False
    err = []
    for ID in ["native", "cartesian"]:
        if ID not in tend.ID:
            continue
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
                dat.name = dat.description[:2] + "forcing"
                ref.name = ref.description
                plotting.scatter_hue(dat, ref, title=log, **plot_kws)
            failed = True
    return failed, min(err)


def test_decomp_sumdir(adv, corr, avg_dims_error=None, thresh=0.99999,
                       loc=None, iloc=None, plot=True, **plot_kws):
    """
    Test that budget methods "native" and "cartesian" give equal advective tendencies
    in all components if the three spatial directions are summed up.

    The test fails if the Nash-Sutcliffe efficiency coefficient (NSE)
    is below the given threshold. If avg_dims_error is given, the averaging in the
    NSE calculation is only carried out over these dimensions. Afterwards the minimum NSE
    value is taken over the remaining dimensions.


    Parameters
    ----------
    adv : xarray DataArray
        Advective tendencies.
    corr : xarray DataArray
        Cartesian corrections for advective tendencies.
    avg_dims_error : str or list of str, optional
        Dimensions over which to calculate the NSE. The default is None.
    thresh : float, optional
        Threshold value for NSE below which the test fails
    loc : dict, optional
        Mapping for label based indexing before running the test. The default is None.
    iloc : dict, optional
        Mapping for integer-location based indexing before running the test. The default is None.
    plot : bool, optional
        Create scatter plot if test fails. The default is True.
    **plot_kws :
        keyword arguments passed to plotting.scatter_hue.

    Returns
    -------
    failed : bool
        Test failed.
    err : float
        Test statistic NSE

    """
    data = adv.sel(dir="sum")
    ref = data.sel(ID="native")
    dat = data.sel(ID="cartesian") + corr.sel(ID="cartesian", dir="T")
    dat = tools.loc_data(dat, loc=loc, iloc=iloc)
    ref = tools.loc_data(ref, loc=loc, iloc=iloc)
    err = tools.nse(dat, ref, dim=avg_dims_error).min().values
    failed = False
    if err < thresh:
        log = "test_decomp_sumdir, {}: min. NSE less than {}: {:.7f}".format(dat.description, thresh, err)
        print(log)
        if plot:
            dat.name = "cartesian"
            ref.name = "native"
            plotting.scatter_hue(dat, ref, title=log, **plot_kws)
        failed = True
    return failed, err


def test_decomp_sumcomp(adv, avg_dims_error=None, thresh=0.9999999999,
                        loc=None, iloc=None, plot=True, **plot_kws):
    """Test that the total advective tendency is indeed the sum of the mean and
    resolved turbulent components in all three spatial directions.

    The test fails if the Nash-Sutcliffe efficiency coefficient (NSE)
    is below the given threshold. If avg_dims_error is given, the averaging in the
    NSE calculation is only carried out over these dimensions. Afterwards the minimum NSE
    value is taken over the remaining dimensions.


    Parameters
    ----------
    adv : xarray DataArray
        Advective tendencies.
    avg_dims_error : str or list of str, optional
        Dimensions over which to calculate the NSE. The default is None.
    thresh : float, optional
        Threshold value for NSE below which the test fails
    loc : dict, optional
        Mapping for label based indexing before running the test. The default is None.
    iloc : dict, optional
        Mapping for integer-location based indexing before running the test. The default is None.
    plot : bool, optional
        Create scatter plot if test fails. The default is True.
    **plot_kws :
        keyword arguments passed to plotting.scatter_hue.

    Returns
    -------
    failed : bool
        Test failed.
    err : float
        Test statistic NSE

    """
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
                ref.name = "adv_r"
                dat.name = "mean + trb_r"
                plotting.scatter_hue(dat, ref, title=log, **plot_kws)
            failed = True
    return failed, min(err)


def test_dz_out(adv, avg_dims_error=None, thresh=0.9, loc=None, iloc=None, plot=True, **plot_kws):
    """Test that the Cartesian corrections imposed by the budget methods
    "cartesian" and "cartesian dz_out_z" lead to
    similar advective tendencies in all three directions and components.

    The test fails if the Nash-Sutcliffe efficiency coefficient (NSE)
    is below the given threshold. If avg_dims_error is given, the averaging in the
    NSE calculation is only carried out over these dimensions. Afterwards the minimum NSE
    value is taken over the remaining dimensions.


    Parameters
    ----------
    adv : xarray DataArray
        Advective tendencies.
    avg_dims_error : str or list of str, optional
        Dimensions over which to calculate the NSE. The default is None.
    thresh : float, optional
        Threshold value for NSE below which the test fails
    loc : dict, optional
        Mapping for label based indexing before running the test. The default is None.
    iloc : dict, optional
        Mapping for integer-location based indexing before running the test. The default is None.
    plot : bool, optional
        Create scatter plot if test fails. The default is True.
    **plot_kws :
        keyword arguments passed to plotting.scatter_hue.

    Returns
    -------
    failed : bool
        Test failed.
    err : float
        Test statistic NSE

    """
    failed = False
    ref = adv.sel(ID="cartesian")
    dat = adv.sel(ID="cartesian dz_out_z")
    dat = tools.loc_data(dat, loc=loc, iloc=iloc)
    ref = tools.loc_data(ref, loc=loc, iloc=iloc)
    err = tools.nse(dat, ref, dim=avg_dims_error).min().values
    if err < thresh:
        log = "test_dz_out, {} (XYZ): min. NSE less than {}: {:.5f}".format(dat.description, thresh, err)
        print(log)
        if plot:
            dat.name = "dz_out_z"
            ref.name = "reference corr."
            plotting.scatter_hue(dat, ref, title=log, **plot_kws)
        failed = True
    return failed, err


def test_2nd(adv, avg_dims_error=None, thresh=0.999, loc=None, iloc=None, plot=True, **plot_kws):
    """Test that the advective tendencies resulting from 2nd-order and
    correct advection order are equal in all three directions and components
    (usually carried out if correct order is equal to 2nd order).

    The test fails if the Nash-Sutcliffe efficiency coefficient (NSE)
    is below the given threshold. If avg_dims_error is given, the averaging in the
    NSE calculation is only carried out over these dimensions. Afterwards the minimum NSE
    value is taken over the remaining dimensions.


    Parameters
    ----------
    adv : xarray DataArray
        Advective tendencies.
    avg_dims_error : str or list of str, optional
        Dimensions over which to calculate the NSE. The default is None.
    thresh : float, optional
        Threshold value for NSE below which the test fails
    loc : dict, optional
        Mapping for label based indexing before running the test. The default is None.
    iloc : dict, optional
        Mapping for integer-location based indexing before running the test. The default is None.
    plot : bool, optional
        Create scatter plot if test fails. The default is True.
    **plot_kws :
        keyword arguments passed to plotting.scatter_hue.

    Returns
    -------
    failed : bool
        Test failed.
    err : float
        Test statistic NSE

    """
    failed = False
    ref = adv.sel(ID="cartesian")
    dat = adv.sel(ID="cartesian 2nd")
    dat = tools.loc_data(dat, loc=loc, iloc=iloc)
    ref = tools.loc_data(ref, loc=loc, iloc=iloc)
    err = tools.nse(dat, ref, dim=avg_dims_error).min().values
    if err < thresh:
        log = "test_2nd, {} (XYZ): min. NSE less than {}: {:.5f}".format(dat.description, thresh, err)
        print(log)
        if plot:
            ref.name = "correct order"
            dat.name = "2nd order"
            plotting.scatter_hue(dat, ref, title=log, **plot_kws)
        failed = True
    return failed, err


def test_w(dat_inst, avg_dims_error=None, thresh=0.995, loc=None, iloc=None, plot=True, **plot_kws):
    """Test that the instantaneous vertical velocity is very similar to the
    instantaneous diagnosed vertical velocity used in the tendency calculations.

    The test fails if the Nash-Sutcliffe efficiency coefficient (NSE)
    is below the given threshold. If avg_dims_error is given, the averaging in the
    NSE calculation is only carried out over these dimensions. Afterwards the minimum NSE
    value is taken over the remaining dimensions.


    Parameters
    ----------
    adv : xarray DataArray
        Advective tendencies.
    avg_dims_error : str or list of str, optional
        Dimensions over which to calculate the NSE. The default is None.
    thresh : float, optional
        Threshold value for NSE below which the test fails
    loc : dict, optional
        Mapping for label based indexing before running the test. The default is None.
    iloc : dict, optional
        Mapping for integer-location based indexing before running the test. The default is None.
    plot : bool, optional
        Create scatter plot if test fails. The default is True.
    **plot_kws :
        keyword arguments passed to plotting.scatter_hue.

    Returns
    -------
    failed : bool
        Test failed.
    err : float
        Test statistic NSE

    """
    dat_inst = tools.loc_data(dat_inst, loc=loc, iloc=iloc)
    ref = dat_inst["W"]
    dat = dat_inst["W_DIAG"]
    err = tools.nse(dat, ref, dim=avg_dims_error).min().values
    failed = False
    if err < thresh:
        log = "test_w: min. NSE less than {}: {:.5f}".format(thresh, err)
        print(log)
        if plot:
            plotting.scatter_hue(dat, ref, title=log, **plot_kws)
        failed = True
    return failed, err


def test_nan(datout, cut_bounds=None):
    """Test for NaN and inf values in postprocessed data.
    If invalid values occur, print reduced dataset to show their locations.

    Parameters
    ----------
    datout : xarray dataset or DataArray
        Input data.
    cut_bounds : iterable, optional
        List of dimensions for which to cut the boundaries before testing.
        For each dimension the staggered and unstaggered version are considered.
        The default is None.

    Returns
    -------
    failed : bool
        Test failed.

    """
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


def test_y0(adv, thresh=(1e-5, 5e-2)):
    """Test whether the advective tendency resulting from fluxes in y-direction is,
    on average, much smaller than the one resulting from fluxes in x-direction. This
    should be the case if the budget is averaged over y. The average absolute ratio is
    compared to the given thresholds for the two budget methods "native" and "cartesian".
    """
    failed = False
    dims = [d for d in adv.dims if d not in ["dir", "ID"]]
    f = abs((adv.sel(dir="Y") / adv.sel(dir="X"))).mean(dims)
    for ID, thresh in zip(["native", "cartesian"], thresh):
        fi = f.loc[ID].values
        if fi > thresh:
            print("test_y0 failed for ID={}!: mean(|adv_y/adv_x|) = {} > {}".format(ID, fi, thresh))
            failed = True
    return failed, f.values


def test_dim_coords(dat, dat_inst, variable, dat_name, failed):
    """
    Test if dimension coordinates in postprocessed output are the same as in instantaneous WRF output.

    Exclude Time, ID, dir, and comp.

    """
    for dim in dat.dims:
        c = dat[dim].values
        if dim in ["Time", "ID", "dir", "comp"]:
            cr = None
        elif dim in dat_inst.dims:
            cr = dat_inst[dim].values
        if cr is not None:
            if (len(c) != len(cr)) or (c != cr).any():
                print("Coordinates for dimension {} in data {} of variable {}"
                      " differs between postprocessed output and WRF output:"
                      "\n {} vs. {}".format(dim, dat_name, variable, c, cr))
                f = "FAIL"
            else:
                f = "pass"
            f0 = failed.loc[variable, "dim_coords"]
            if (f0 == "") or (f0 == "pass"):
                failed.loc[variable, "dim_coords"] = f


def test_no_model_change(outpath, ID, inst_file, mean_file):
    """
    Check that output of WRFlux and original WRF is identical for all history variables.

    Parameters
    ----------
    outpath : str
        Path to the WRF output directory.
    ID : str
        Simulation ID with placeholder for build.
    inst_file : str or path-like
        Name of the output file containing instantaneous data.
    mean_file : str
        Name of the output file containing time-averaged data.

    Returns
    -------
    res : str
        "FAIL" or "pass".

    """
    res = "pass"
    # open WRF output
    dat_inst = {}
    for build in ["org", "debug"]:
        outpath_c = os.path.join(outpath, ID.format(build)) + "_0"
        _, dat_inst[build] = tools.load_data(outpath_c, inst_file=inst_file, mean_file=mean_file)

    # check that the right output was loaded
    assert "W_DIAG" in dat_inst["debug"]
    assert "W_DIAG" not in dat_inst["org"]

    for v in dat_inst["org"].variables:
        try:
            xr.testing.assert_identical(dat_inst["debug"][v], dat_inst["org"][v])
        except AssertionError:
            print("Simulation with WRFlux and original WRF differ in variable {}!".format(v))
            res = "FAIL"
    return res


# %% run_tests

def run_tests(datout, tests, dat_inst=None, sim_id=None, trb_exp=False,
              hor_avg=False, chunks=None, **kw):
    """Run test functions for WRF output postprocessed with WRFlux.
       Thresholds are hard-coded.

    Parameters
    ----------
    datout : nested dict
        Postprocessed output for all variables.
    tests : list of str
        Tests to perform.
        Choices: budget, decomp_sumdir, decomp_sumcomp, dz_out, adv_2nd, w, Y=0, NaN
    dat_inst : xarray DataArray, optional
        WRF instantaneous output needed for w test. The default is None.
    sim_id : str, optional
        ID of the current test simulation. The default is None.
    trb_exp : bool, optional
        Turbulent fluxes were calculated explicitly. The default is False.
    hor_avg : bool, optional
        Horizontal averaging was used in postprocessing. The default is False.
    chunks : dict of integers, optional
        Mapping from dimension "x" and/or "y" to chunk sizes used in postprocessing.
        If given, the boundaries in the chunking directions are pruned.
        The default is None.
    **kw :
        Keyword arguments passed to test functions.

    Returns
    -------
    failed : pandas DataFrame
        "FAIL" and "pass" labels for all tests and variables.
    err : pandas DataFrame
        NSE error statistics for performed tests.

    """
    if tests is None:
        tests = all_tests

    variables = list(datout.keys())
    failed = pd.DataFrame(columns=tests, index=variables)
    err = pd.DataFrame(columns=tests, index=variables)
    failed[:] = ""
    err[:] = ""

    # cut boundaries for non-periodic BC or if chunking was used
    attrs = datout[variables[0]]["flux"].attrs
    iloc = {}
    if (not attrs["PERIODIC_X"]) or (chunks is not None and "x" in chunks):
        iloc["x"] = slice(1, -1)
    if (not attrs["PERIODIC_Y"]) or (chunks is not None and "y" in chunks):
        iloc["y"] = slice(1, -1)

    if attrs["PERIODIC_Y"] == 0:
        if "Y=0" in tests:
            tests.remove("Y=0")
    # for w test: cut first time step
    dat_inst_lim = dat_inst.isel(Time=slice(1, None), **iloc)

    datout_lim = {}
    for v, datout_v in datout.items():
        datout_lim[v] = {}
        for n, dat in datout_v.items():
            if "dim_coords" in tests:
                test_dim_coords(dat, dat_inst, v, n, failed)
            datout_lim[v][n] = tools.loc_data(dat, iloc=iloc)

    fpath = Path(__file__).parent
    for var, datout_v in datout_lim.items():
        print("Variable: " + var)
        figloc = fpath / "figures" / var
        failed_i = {}
        err_i = {}

        if "budget" in tests:
            # TODOm: change threshold depending on ID?
            tend = datout_v["tend"].sel(comp="tendency")
            forcing = datout_v["tend"].sel(comp="forcing")
            kw["figloc"] = figloc / "budget"
            if (var == "w") and (sim_id is not None) and ("open BC y hor_avg" in sim_id):
                kw["thresh"] = 0.99
            elif (var == "t") and (attrs["USE_THETA_M"] == 1) and (attrs["OUTPUT_DRY_THETA_FLUXES"] == 1):
                # reduce threshold for WENO and monotonic advection as
                # dry theta budget is not perfectly closed
                if (attrs["SCALAR_ADV_OPT"] >= 3) and (attrs["MOIST_ADV_OPT"] >= 3):
                    kw["thresh"] = 0.95
                elif (attrs["SCALAR_ADV_OPT"] >= 3):
                    kw["thresh"] = 0.7
                elif attrs["MOIST_ADV_OPT"] == 2:
                    kw["thresh"] = 0.98

            failed_i["budget"], err_i["budget"] = test_budget(tend, forcing, **kw)
            if "thresh" in kw:
                del kw["thresh"]
        adv = datout_v["adv"]
        if "decomp_sumdir" in tests:
            if trb_exp or hor_avg or (attrs["HESSELBERG_AVG"] == 0):
                # reduce threshold
                kw["thresh"] = 0.992
            kw["figloc"] = figloc / "decomp_sumdir"
            failed_i["decomp_sumdir"], err_i["decomp_sumdir"] = test_decomp_sumdir(
                adv, datout_v["corr"], **kw)
            if "thresh" in kw:
                del kw["thresh"]

        if "decomp_sumcomp" in tests:
            if trb_exp:
                # reduce threshold for explicit turbulent fluxes
                kw["thresh"] = 0.995
            kw["figloc"] = figloc / "decomp_sumcomp"
            failed_i["decomp_sumcomp"], err_i["decomp_sumcomp"] = test_decomp_sumcomp(adv, **kw)
            if "thresh" in kw:
                del kw["thresh"]

        if ("dz_out" in tests) and (var != "q"):  # TODOm: why so bad for q?
            kw["figloc"] = figloc / "dz_out"
            failed_i["dz_out"], err_i["dz_out"] = test_dz_out(adv, **kw)

        if "adv_2nd" in tests:
            kw["figloc"] = figloc / "adv_2nd"
            failed_i["adv_2nd"], err_i["adv_2nd"] = test_2nd(adv, **kw)

        if ("w" in tests) and (var == variables[-1]) and (dat_inst is not None):
            # only do test once: for last variable
            kw["figloc"] = figloc / "w"
            failed_i["w"], err_i["w"] = test_w(dat_inst_lim, **kw)

        if "NaN" in tests:
            failed_i["NaN"] = test_nan(datout_v)

        if hor_avg and ("Y=0" in tests):
            failed_i["Y=0"], err_i["Y=0"] = test_y0(adv)

        # store results
        for test, f in failed_i.items():
            if f:
                failed.loc[var, test] = "FAIL"
            else:
                failed.loc[var, test] = "pass"

        for test, e in err_i.items():
            err.loc[var, test] = e

    return failed, err


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
