#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for automatic testing of WRFlux output.

@author: Matthias Göbel
"""
from wrflux import tools, plotting
import pandas as pd
import numpy as np
import xarray as xr
import os
from pathlib import Path
from functools import partial

all_tests = ["budget", "decomp_sumdir", "decomp_sumcomp",
             "dz_out", "adv_2nd", "w", "mass", "Y=0", "NaN", "dim_coords",
             "no_model_change"]


# %% test functions

def test_budget(tend, forcing, avg_dims_error=None, thresh=0.9999,
                loc=None, iloc=None, plot=True, **plot_kws):
    """
    Test closure of budget: tend = forcing.

    Only the budget methods "native" and "cartesian" are tested.
    The test fails if the coefficient of determination
    is below the given threshold. If avg_dims_error is given, the averaging in the
    R2 calculation is only carried out over these dimensions. Afterwards the minimum R2
    value is taken over the remaining dimensions.

    Parameters
    ----------
    tend : xarray DataArray
        Total tendency.
    forcing : xarray DataArray
        Total forcing.
    avg_dims_error : str or list of str, optional
        Dimensions over which to calculate the R2. The default is None.
    thresh : float, optional
        Threshold value for R2 below which the test fails
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
        Test statistic R2

    """
    failed = False
    err = []
    fname = None
    if "fname" in plot_kws:
        fname = plot_kws.pop("fname")
    for ID in ["native", "cartesian"]:
        if ID not in tend.ID:
            continue
        ref = tend.sel(ID=ID, drop=True)
        dat = forcing.sel(ID=ID, drop=True)
        dat = tools.loc_data(dat, loc=loc, iloc=iloc)
        ref = tools.loc_data(ref, loc=loc, iloc=iloc)
        e = R2(dat, ref, dim=avg_dims_error).min().values
        err.append(e)

        if e < thresh:
            log = "test_budget for ID='{}': min. R2 less than {}: {:.7f}\n".format(ID, thresh, e)
            print(log)
            if plot:
                dat.name = dat.description[:2] + "forcing"
                ref.name = ref.description
                fname_i = fname
                if fname is not None:
                    fname_i = "ID=" + ID + "_" + fname
                plotting.scatter_hue(dat, ref, title=log, fname=fname_i, **plot_kws)
            failed = True

    return failed, min(err)


def test_decomp_sumdir(adv, corr, avg_dims_error=None, thresh=0.99999,
                       loc=None, iloc=None, plot=True, **plot_kws):
    """
    Test that budget methods "native" and "cartesian" give equal advective tendencies
    in all components if the three spatial directions are summed up.

    The test fails if the coefficient of determination
    is below the given threshold. If avg_dims_error is given, the averaging in the
    R2 calculation is only carried out over these dimensions. Afterwards the minimum R2
    value is taken over the remaining dimensions.


    Parameters
    ----------
    adv : xarray DataArray
        Advective tendencies.
    corr : xarray DataArray
        Cartesian corrections for advective tendencies.
    avg_dims_error : str or list of str, optional
        Dimensions over which to calculate the R2. The default is None.
    thresh : float, optional
        Threshold value for R2 below which the test fails
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
        Test statistic R2

    """
    data = adv.sel(dir="sum")
    ref = data.sel(ID="native")
    dat = data.sel(ID="cartesian") - corr.sel(ID="cartesian", dir="T")
    dat = tools.loc_data(dat, loc=loc, iloc=iloc)
    ref = tools.loc_data(ref, loc=loc, iloc=iloc)
    err = R2(dat, ref, dim=avg_dims_error).min().values
    failed = False
    if err < thresh:
        log = "test_decomp_sumdir, {}: min. R2 less than {}: {:.7f}".format(dat.description, thresh, err)
        print(log)
        if plot:
            dat.name = "cartesian"
            ref.name = "native"
            plotting.scatter_hue(dat, ref, title=log, **plot_kws)
        failed = True
    return failed, err


def test_decomp_sumcomp(adv, avg_dims_error=None, thresh=0.999995,
                        loc=None, iloc=None, plot=True, **plot_kws):
    """Test that the total advective tendency is indeed the sum of the mean and
    resolved turbulent components in all three spatial directions.

    The test fails if the coefficient of determination
    is below the given threshold. If avg_dims_error is given, the averaging in the
    R2 calculation is only carried out over these dimensions. Afterwards the minimum R2
    value is taken over the remaining dimensions.


    Parameters
    ----------
    adv : xarray DataArray
        Advective tendencies.
    avg_dims_error : str or list of str, optional
        Dimensions over which to calculate the R2. The default is None.
    thresh : float, optional
        Threshold value for R2 below which the test fails
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
        Test statistic R2

    """
    ref = adv.sel(comp="trb_r")
    dat = adv.sel(comp="adv_r") - adv.sel(comp="mean")
    dat = tools.loc_data(dat, loc=loc, iloc=iloc)
    ref = tools.loc_data(ref, loc=loc, iloc=iloc)
    failed = False
    err = []
    fname = None
    if "fname" in plot_kws:
        fname = plot_kws.pop("fname")
    for ID in dat.ID.values:
        dat_i = dat.sel(ID=ID)
        ref_i = ref.sel(ID=ID)
        e = R2(dat_i, ref_i, dim=avg_dims_error).min().values
        err.append(e)
        if e < thresh:
            log = "decomp_sumcomp, {} (XYZ) for ID={}: min. R2 less than {}: {:.8f}".format(
                dat.description, ID, thresh, e)
            print(log)
            if plot:
                ref_i.name = "trb_r"
                dat_i.name = "adv_r - mean"
                fname_i = fname
                if fname is not None:
                    fname_i = "ID=" + ID + "_" + fname
                plotting.scatter_hue(dat_i, ref_i, title=log, fname=fname_i, **plot_kws)
            failed = True
    return failed, min(err)


def test_dz_out(adv, avg_dims_error=None, thresh=0.95, loc=None, iloc=None, plot=True, **plot_kws):
    """Test that the Cartesian corrections imposed by the budget methods
    "cartesian" and "cartesian dz_out_z" lead to
    similar advective tendencies in all three directions and components.

    The test fails if the coefficient of determination
    is below the given threshold. If avg_dims_error is given, the averaging in the
    R2 calculation is only carried out over these dimensions. Afterwards the minimum R2
    value is taken over the remaining dimensions.


    Parameters
    ----------
    adv : xarray DataArray
        Advective tendencies.
    avg_dims_error : str or list of str, optional
        Dimensions over which to calculate the R2. The default is None.
    thresh : float, optional
        Threshold value for R2 below which the test fails
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
        Test statistic R2

    """
    failed = False
    ref = adv.sel(ID="cartesian")
    dat = adv.sel(ID="cartesian dz_out_z")
    dat = tools.loc_data(dat, loc=loc, iloc=iloc)
    ref = tools.loc_data(ref, loc=loc, iloc=iloc)
    err = R2(dat, ref, dim=avg_dims_error).min().values
    if err < thresh:
        log = "test_dz_out, {} (XYZ): min. R2 less than {}: {:.5f}".format(dat.description, thresh, err)
        print(log)
        if plot:
            dat.name = "dz_out_z"
            ref.name = "reference corr."
            plotting.scatter_hue(dat, ref, title=log, **plot_kws)
        failed = True
    return failed, err


def test_2nd(adv, avg_dims_error=None, thresh=0.998, loc=None, iloc=None, plot=True, **plot_kws):
    """Test that the advective tendencies resulting from 2nd-order and
    correct advection order are equal in all three directions and components
    (usually carried out if correct order is equal to 2nd order).

    The test fails if the coefficient of determination
    is below the given threshold. If avg_dims_error is given, the averaging in the
    R2 calculation is only carried out over these dimensions. Afterwards the minimum R2
    value is taken over the remaining dimensions.


    Parameters
    ----------
    adv : xarray DataArray
        Advective tendencies.
    avg_dims_error : str or list of str, optional
        Dimensions over which to calculate the R2. The default is None.
    thresh : float, optional
        Threshold value for R2 below which the test fails
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
        Test statistic R2

    """
    failed = False
    ref = adv.sel(ID="cartesian")
    dat = adv.sel(ID="cartesian 2nd")
    dat = tools.loc_data(dat, loc=loc, iloc=iloc)
    ref = tools.loc_data(ref, loc=loc, iloc=iloc)
    err = R2(dat, ref, dim=avg_dims_error).min().values
    if err < thresh:
        log = "test_2nd, {} (XYZ): min. R2 less than {}: {:.5f}".format(dat.description, thresh, err)
        print(log)
        if plot:
            ref.name = "correct order"
            dat.name = "2nd order"
            plotting.scatter_hue(dat, ref, title=log, **plot_kws)
        failed = True
    return failed, err


def test_w(dat_inst, avg_dims_error=None, thresh=0.9995, loc=None, iloc=None, plot=True, **plot_kws):
    """Test that the instantaneous vertical velocity is very similar to the
    instantaneous diagnosed vertical velocity used in the tendency calculations.

    The test fails if the coefficient of determination
    is below the given threshold. If avg_dims_error is given, the averaging in the
    R2 calculation is only carried out over these dimensions. Afterwards the minimum R2
    value is taken over the remaining dimensions.


    Parameters
    ----------
    adv : xarray DataArray
        Advective tendencies.
    avg_dims_error : str or list of str, optional
        Dimensions over which to calculate the R2. The default is None.
    thresh : float, optional
        Threshold value for R2 below which the test fails
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
        Test statistic R2

    """
    dat_inst = tools.loc_data(dat_inst, loc=loc, iloc=iloc)
    ref = dat_inst["W_SAVE"]
    dat = dat_inst["W_DIAG"]
    err = R2(dat, ref, dim=avg_dims_error).min().values
    failed = False
    if err < thresh:
        log = "test_w: min. R2 less than {}: {:.6f}".format(thresh, err)
        print(log)
        if plot:
            plotting.scatter_hue(dat, ref, title=log, **plot_kws)
        failed = True
    return failed, err


def test_mass(tend_mass, avg_dims_error=None, thresh=0.99999999,
              loc=None, iloc=None, plot=True, **plot_kws):
    """Test closure of continuity equation.

    In the tendency calculations the vertical component of the continuity equation
    is calculated as residual to improve the budget closure which leads to automatic
    closure of the continuity equation.
    This test ensures that this residual calculation does not produce larger changes
    in the vertical component by comparing the residual calculation with the
    explicit calculation which uses the vertical velocity.
    For the dz_out type formulations, the continuity equation cannot be well closed.
    Therefore, we only compare the individual components with the standard Cartesian
    formulation.
    The test fails if the coefficient of determination
    is below the given threshold. If avg_dims_error is given, the averaging in the
    R2 calculation is only carried out over these dimensions. Afterwards the minimum R2
    value is taken over the remaining dimensions.

    Parameters
    ----------
    tend_mass : xarray DataArray
        Components of continuity equation.
    avg_dims_error : str or list of str, optional
        Dimensions over which to calculate the R2. The default is None.
    thresh : float, optional
        Threshold value for R2 below which the test fails
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
        Test statistic R2

    """
    ref = tend_mass.sel(dir="Z")
    dat = tend_mass.sel(dir="T") - tend_mass.sel(dir="X") - tend_mass.sel(dir="Y")
    failed = False
    err = []
    fname = None
    if "fname" in plot_kws:
        fname = plot_kws.pop("fname")
    for ID in dat.ID.values:
        dat_i = dat.sel(ID=ID)
        ref_i = ref.sel(ID=ID)
        dat_i = tools.loc_data(dat_i, loc=loc, iloc=iloc)
        ref_i = tools.loc_data(ref_i, loc=loc, iloc=iloc)
        e = R2(dat_i, ref_i, dim=avg_dims_error).min().values
        err.append(e)
        if e < thresh:
            log = "test_mass: vertical component of continuity equation\n for ID={}: min. R2 less than {}: {:.10f}".format(ID, thresh, e)
            print(log)
            if plot:
                dat_i.name = "Residual calculation"
                ref_i.name = "Calculation with vertical velocity"

                fname_i = fname
                if fname is not None:
                    fname_i = "ID=" + ID + "_" + fname
                plotting.scatter_hue(dat_i, ref_i, title=log, fname=fname_i, **plot_kws)
            failed = True
    return failed, min(err)


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
            dnan = find_bad(d[dv])
            if (dnan is not None) and (sum(dnan.shape) != 0):
                print("\nWARNING: found NaNs in {} :\n{}".format(v, dnan.coords))
                failed = True

    return failed


def test_y0(adv, thresh=(5e-6, 5e-3)):
    """Test whether the advective tendency resulting from fluxes in y-direction is,
    on average, much smaller than the one resulting from fluxes in x-direction. This
    should be the case if the budget is averaged over y. The average absolute ratio is
    compared to the given thresholds for the two budget methods "native" and "cartesian".
    """
    failed = False
    dims = [d for d in adv.dims if d not in ["dir", "ID", "comp"]]
    f = abs((adv.sel(dir="Y") / adv.sel(dir="X"))).median(dims)
    for ID, thresh_i in zip(["native", "cartesian"], thresh):
        for comp in f.comp.values:
            fi = f.sel(ID=ID, comp=comp).values
            if fi > thresh_i:
                print("test_y0 failed for ID={}, comp={}!: median(|adv_y/adv_x|) = {} > {}".format(ID, comp, fi, thresh_i))
                failed = True
    return failed, f.max("comp").values


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
    assert "OUTPUT_DRY_THETA_FLUXES" in dat_inst["debug"].attrs
    assert "OUTPUT_DRY_THETA_FLUXES" not in dat_inst["org"].attrs

    for v in dat_inst["org"].variables:
        try:
            xr.testing.assert_identical(dat_inst["debug"][v], dat_inst["org"][v])
        except AssertionError:
            print("Simulation with WRFlux and original WRF differ in variable {}!".format(v))
            res = "FAIL"
    return res


# %% run_tests

def run_tests(datout, tests, dat_inst=None, sim_id="", trb_exp=False,
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
        ID of the current test simulation. The default is "".
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
        R2 error statistics for performed tests.

    """
    if tests is None:
        tests = all_tests
    tests = tests.copy()
    for test in tests:
        if test not in all_tests:
            raise ValueError("Test {} not available! Available tests:\n{}".format(test, ", ".join(all_tests)))
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

    avg_dims = None
    if hor_avg:
        avg_dims = []
        dat = datout[variables[0]]["adv"]
        for d in tools.xy:
            if (d not in dat.dims) and (d + "_stag" not in dat.dims):
                avg_dims.append(d)

    # for w test: cut first time step
    dat_inst_lim = dat_inst.isel(Time=slice(1, None), **iloc)

    datout_lim = {}
    for v, datout_v in datout.items():
        datout_lim[v] = {}
        for n, dat in datout_v.items():
            if "ID" in dat.dims:
                # remove theta_pert label from budget method IDs if this does not lead to duplicate labels
                IDs = []
                for ID in dat.ID.values:
                    ID = ID.split(" ")
                    if "theta_pert" in ID:
                        ID_new = ID.copy()
                        ID_new.remove("theta_pert")
                        if len(ID_new) == 0:
                            ID_new = ["native"]
                        if ID_new not in dat.ID:
                            ID = ID_new
                    IDs.append(" ".join(ID))
                dat["ID"] = IDs
            if "dim_coords" in tests:
                test_dim_coords(dat, dat_inst, v, n, failed)
            if hor_avg:
                for avg_dim in avg_dims:
                    for stag in ["", "_stag"]:
                        assert avg_dim + stag not in dat.dims
            datout_lim[v][n] = tools.loc_data(dat, iloc=iloc)

    fpath = Path(__file__).parent
    for var, datout_v in datout_lim.items():
        print("Variable: " + var)
        figloc = fpath / "figures" / var
        failed_i = {}
        err_i = {}

        if "budget" in tests:
            tend = datout_v["tend"].sel(comp="tendency")
            forcing = datout_v["tend"].sel(comp="forcing")
            kw["figloc"] = figloc / "budget"
            if (var == "w") and ("open BC y hor_avg" in sim_id):
                kw["thresh"] = 0.995
            elif (var in ["u", "v", "w"]) and ("open BC" in sim_id):
                kw["thresh"] = 0.999
            elif var == "t":
                if "open BC" in sim_id:
                    kw["thresh"] = 0.999
                if "symmetric BC" in sim_id:
                    kw["thresh"] = 0.995
                elif attrs["USE_THETA_M"] == 1:

                    if attrs["OUTPUT_DRY_THETA_FLUXES"] == 0:
                        # lower thresh as cartesian tendency for thm is close to 0
                        if attrs["MP_PHYSICS"] > 0:
                            kw["thresh"] = 0.96
                        else:
                            kw["thresh"] = 0.995

                    # reduce threshold for WENO and monotonic advection as
                    # dry theta budget is not perfectly closed
                    elif (attrs["SCALAR_ADV_OPT"] >= 3) and (attrs["MOIST_ADV_OPT"] >= 3):
                        kw["thresh"] = 0.94
                    elif (attrs["SCALAR_ADV_OPT"] >= 3):
                        kw["thresh"] = 0.8
                    elif attrs["MOIST_ADV_OPT"] == 2:
                        kw["thresh"] = 0.97

            failed_i["budget"], err_i["budget"] = test_budget(tend, forcing, **kw)
            if "thresh" in kw:
                del kw["thresh"]
        adv = datout_v["adv"]
        if "decomp_sumdir" in tests:
            if attrs["HESSELBERG_AVG"] == 0:
                kw["thresh"] = 0.995
            elif trb_exp:
                kw["thresh"] = 0.999
            kw["figloc"] = figloc / "decomp_sumdir"
            failed_i["decomp_sumdir"], err_i["decomp_sumdir"] = test_decomp_sumdir(
                adv, datout_v["corr"], **kw)
            if "thresh" in kw:
                del kw["thresh"]

        if "decomp_sumcomp" in tests:
            if trb_exp:
                # reduce threshold for explicit turbulent fluxes
                kw["thresh"] = 0.999
            kw["figloc"] = figloc / "decomp_sumcomp"
            failed_i["decomp_sumcomp"], err_i["decomp_sumcomp"] = test_decomp_sumcomp(adv, **kw)
            if "thresh" in kw:
                del kw["thresh"]

        if ("dz_out" in tests) and (var != "q"):  # TODOm: why so bad for q?
            kw["figloc"] = figloc / "dz_out"
            adv_noavgdir = adv
            if hor_avg:
                thresh = {"t": 0.85, "u": 0.7, "v": 0.995, "w": 0.92}
                kw["thresh"] = thresh[var]
                adv_noavgdir = adv.sel(dir=[d for d in adv.dir.values if d.lower() not in avg_dims])
            failed_i["dz_out"], err_i["dz_out"] = test_dz_out(adv_noavgdir, **kw)
            if "thresh" in kw:
                del kw["thresh"]
        if "adv_2nd" in tests:
            kw["figloc"] = figloc / "adv_2nd"
            failed_i["adv_2nd"], err_i["adv_2nd"] = test_2nd(adv, **kw)
        if ("w" in tests) and (var == variables[-1]) and (dat_inst is not None):
            # only do test once: for last variable
            kw["figloc"] = figloc / "w"
            failed_i["w"], err_i["w"] = test_w(dat_inst_lim, **kw)

        if ("mass" in tests) and (var == "t"):
            if "dz_out" in tests:
                if hor_avg:
                    kw["thresh"] = 0.85
                else:
                    kw["thresh"] = 0.995

            elif attrs["HESSELBERG_AVG"] == 0:
                kw["thresh"] = 0.99999

            kw["figloc"] = figloc / "mass"
            failed_i["mass"], err_i["mass"] = test_mass(datout_v["tend_mass"], **kw)
            if "thresh" in kw:
                del kw["thresh"]

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


# %% other functions

def dropna_dims(dat, dims=None, how="all", **kwargs):
    """
    Consecutively drop NaNs along given dimensions.

    Parameters
    ----------
    dat : xarray dataset or dataarray
        input data.
    dims : iterable, optional
        dimensions to use. The default is None, which takes all dimensions.
    how : str, optional
        drop index if "all" or "any" NaNs occur. The default is "all".
    **kwargs : keyword arguments
        kwargs for dropna.

    Returns
    -------
    dat : xarray dataset or dataarray
        reduced data.

    """
    if dims is None:
        dims = dat.dims
    for d in dims:
        dat = dat.dropna(d, how=how, **kwargs)

    return dat


def find_bad(dat, nan=True, inf=True):
    """Drop all indeces of each dimension in DataArray dat that do not contain any NaNs or infs."""
    # set coordinates for all dims
    for d in dat.dims:
        if d not in dat.coords:
            dat = dat.assign_coords({d: dat[d]})

    nans = False
    infs = False
    if nan:
        nans = dat.isnull()
    if inf:
        infs = dat == np.inf
    invalid = nans | infs
    invalid = invalid.where(invalid)
    invalid = dropna_dims(invalid)
    if invalid.size > 0:
        dat = dat.loc[invalid.indexes]
    else:
        dat = None
    return dat


def R2(dat, ref, dim=None):
    """
    Coefficient of determination for input data with respect to reference data.

    Parameters
    ----------
    dat : datarray
        input data.
    ref : datarray
        reference data.
    dim : str or list, optional
        dimensions along which to calculate the index.
        The default is None, which means all dimensions.

    Returns
    -------
    datarray
        R2.

    """
    if dim is not None:
        dim = tools.make_list(dim)
        d = dict(dim=tools.correct_dims_stag_list(dim, ref))
    else:
        d = {}
    dat = dat.astype(np.float64)
    ref = ref.astype(np.float64)
    mse = ((dat - ref)**2).mean(**d)
    var = ((ref - ref.mean(**d))**2).mean(**d)
    return 1 - mse / var


def trb_fluxes(dat_mean, inst, variables, grid, t_avg_interval,
               cyclic=None, hor_avg=False, avg_dims=None):
    """Compute turbulent fluxes explicitly from complete timeseries output.

    Parameters
    ----------
    dat_mean : xarray Dataset
        WRF time-averaged output.
    inst : xarray Dataset
        WRF flux output at every time step.
    variables : list of str
        List of variables to process.
    grid : xarray Dataset
        Variables related to the model grid.
    t_avg_interval : integer
        Interval for time averaging (number of output time steps) if t_avg=True.
        The default is None.
    cyclic : dict of booleans for xy or None, optional
        Defines which dimensions have periodic boundary conditions.
        Use periodic boundary conditions to fill lateral boundary points.
        The default is None.
    hor_avg : bool, optional
        Average horizontally. The default is False.
    avg_dims : str or list of str, optional
        Averaging dimensions if hor_avg=True. The default is None.

    Returns
    -------
    None.

    """
    avg_kwargs = {"Time": t_avg_interval, "coord_func": {
        "Time": partial(tools.select_ind, indeces=-1)}, "boundary": "trim"}

    # define all needed variables
    all_vars = ["OMZN_MEAN"]
    for var in variables:
        for d, vel in zip(tools.XYZ, tools.uvw):
            all_vars.append(var.upper() + d + "_MEAN")
            all_vars.append(vel.upper() + "_MEAN")

    # fill all time steps with block average
    means = dat_mean[all_vars].reindex(Time=inst.Time).bfill("Time")

    for var in variables:
        var = var.upper()
        for d, v in zip(["X", "Y", "Z", "Z"], ["U", "V", "W", "OMZN"]):
            var_d = var + d + "_MEAN"
            vel = v + "_MEAN"

            if d in ["X", "Y"]:
                rho = tools.build_mu(inst["MUT_MEAN"], grid, full_levels="bottom_top_stag" in inst[var_d].dims)
            else:
                rho = inst["RHOD_MEAN"]

            var_d_m = means[var_d]
            vel_m = means[vel]
            if hor_avg:
                var_d_m = tools.avg_xy(var_d_m, avg_dims, rho=rho, cyclic=cyclic, **grid[tools.stagger_const])
                vel_m = tools.avg_xy(vel_m, avg_dims, rho=rho, cyclic=cyclic, **grid[tools.stagger_const])

            # compute perturbations
            var_pert = inst[var_d] - var_d_m
            rho_stag_vel = tools.stagger_like(rho, inst[vel],
                                              cyclic=cyclic, **grid[tools.stagger_const])
            vel_pert = tools.stagger_like(rho_stag_vel * (inst[vel] - vel_m), var_pert,
                                          cyclic=cyclic, **grid[tools.stagger_const])
            # build flux
            flux = vel_pert * var_pert
            flux = flux.coarsen(**avg_kwargs).mean()
            if hor_avg and (d.lower() not in avg_dims):
                flux = tools.avg_xy(flux, avg_dims, cyclic=cyclic)
                rho = tools.avg_xy(rho, avg_dims, cyclic=cyclic, **grid[tools.stagger_const])

            rho_stag = tools.stagger_like(rho, var_pert, cyclic=cyclic,
                                          **grid[tools.stagger_const])
            rho_stag_mean = rho_stag.coarsen(**avg_kwargs).mean()
            flux = flux / rho_stag_mean
            dat_mean["F{}{}_TRB_MEAN".format(var, v)] = flux

            # rho_stag_vel_mean = rho_stag_vel.coarsen(**avg_kwargs).mean()
            # vel_stag = tools.stagger_like(rho_stag_vel * inst[vel], var_pert,
            #                               cyclic=cyclic, **grid[tools.stagger_const])
            # vel_stag_mean = tools.stagger_like(rho_stag_vel_mean * dat_mean[vel], var_pert,
            #                                    cyclic=cyclic, **grid[tools.stagger_const])
            # tot_flux = vel_stag * inst[var_d]
            # tot_flux = tot_flux.coarsen(**avg_kwargs).mean()
            # mean_flux = vel_stag_mean * dat_mean[var_d]
            # if hor_avg and (d.lower() not in avg_dims):
            # tot_flux = tools.avg_xy(tot_flux, avg_dims, cyclic=cyclic)
            # mean_flux = tools.avg_xy(mean_flux, avg_dims, cyclic=cyclic)
            # tot_flux = tot_flux / rho_stag_mean
            # dat_mean["F{}{}_TOT_MEAN".format(var, v)] = tot_flux
            # mean_flux = mean_flux / rho_stag_mean
            # dat_mean["F{}{}_MEAN_MEAN".format(var, v)] = mean_flux

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
