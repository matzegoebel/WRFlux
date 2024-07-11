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

all_tests = [
    "budget",
    "decomp_sumdir",
    "decomp_sumcomp",
    "sgs",
    "w",
    "mass",
    "Y=0",
    "NaN",
    "dim_coords",
    "no_model_change",
    "periodic",
    "adv_form",
]


# %% test functions


def test_budget(
    tend,
    forcing,
    avg_dims_error=None,
    thresh=0.9998,
    thresh_cartesian=None,
    budget_forms=("native", "adv_form", "cartesian", "cartesian adv_form"),
    loc=None,
    iloc=None,
    plot=True,
    **plot_kws,
):
    """
    Test closure of budget: tend = forcing.

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
    thresh_cartesian : float, optional
        Use different threshold value for Cartesian coordinate system.
        The default is None, for which 'thresh' is used in both formulations.
    budget_forms : list of str
        Budget forms to consider.
        By default, only "native", "adv_form", "cartesian", and "cartesian adv_form" are tested.
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
    fname = ""
    if "fname" in plot_kws:
        fname = plot_kws.pop("fname")
    for budget_form in budget_forms:
        thresh_i = thresh
        if (budget_form == "cartesian") and (thresh_cartesian is not None):
            thresh_i = thresh_cartesian
        if budget_form not in tend.budget_form:
            continue
        ref = tend.sel(budget_form=budget_form, drop=True)
        dat = forcing.sel(budget_form=budget_form, drop=True)
        dat = tools.loc_data(dat, loc=loc, iloc=iloc)
        ref = tools.loc_data(ref, loc=loc, iloc=iloc)
        e = R2(dat, ref, dim=avg_dims_error).min().values
        err.append(e)

        if e < thresh_i:
            log = "test_budget for budget_form='{}': min. R2 less than {}: {:.10f}\n".format(
                budget_form, thresh_i, e
            )
            print(log)
            if plot:
                dat.name = dat.description[:8] + "forcing"
                ref.name = ref.description
                fname_i = fname
                if fname is not None:
                    fname_i = "budget_form=" + budget_form + "_" + fname
                    log = fname_i + "\n" + log
                plotting.scatter_hue(dat, ref, title=log, fname=fname_i, **plot_kws)
            failed = True

    return failed, min(err)


def test_decomp_sumdir(
    adv, corr, avg_dims_error=None, thresh=0.99999, loc=None, iloc=None, plot=True, **plot_kws
):
    """
    Test that budget forms "native" and "cartesian" give equal advective tendencies
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
    data = adv.sel(dir="sum", comp=corr.comp)
    ref = data.sel(budget_form="native")
    dat = data.sel(budget_form="cartesian") - corr.sel(budget_form="cartesian", dir="T")
    dat = tools.loc_data(dat, loc=loc, iloc=iloc)
    ref = tools.loc_data(ref, loc=loc, iloc=iloc)
    err = R2(dat, ref, dim=avg_dims_error).min().values
    failed = False
    if err < thresh:
        log = "test_decomp_sumdir, {}: min. R2 less than {}: {:.7f}".format(
            dat.description, thresh, err
        )
        print(log)
        if plot:
            dat.name = "cartesian"
            ref.name = "native"
            plotting.scatter_hue(dat, ref, title=log, **plot_kws)
        failed = True
    return failed, err


def test_decomp_sumcomp(
    adv, avg_dims_error=None, thresh=0.999995, loc=None, iloc=None, plot=True, **plot_kws
):
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
    dat = adv.sel(comp="total") - adv.sel(comp="mean") - adv.sel(comp="trb_s")
    dat = tools.loc_data(dat, loc=loc, iloc=iloc)
    ref = tools.loc_data(ref, loc=loc, iloc=iloc)
    failed = False
    err = []
    fname = ""
    if "fname" in plot_kws:
        fname = plot_kws.pop("fname")
    for budget_form in dat.budget_form.values:
        dat_i = dat.sel(budget_form=budget_form)
        ref_i = ref.sel(budget_form=budget_form)
        e = R2(dat_i, ref_i, dim=avg_dims_error).min().values
        err.append(e)
        if e < thresh:
            log = (
                "decomp_sumcomp, {} (XYZ) for budget_form={}: min. R2 less than {}: {:.8f}".format(
                    dat.description, budget_form, thresh, e
                )
            )
            print(log)
            if plot:
                ref_i.name = "trb_r"
                dat_i.name = "adv_r - mean"
                fname_i = fname
                if fname is not None:
                    fname_i = "budget_form=" + budget_form + "_" + fname
                    log = fname_i + "\n" + log
                plotting.scatter_hue(dat_i, ref_i, title=log, fname=fname_i, **plot_kws)
            failed = True
    return failed, min(err)


def test_w(
    dat_inst, avg_dims_error=None, thresh=0.9995, loc=None, iloc=None, plot=True, **plot_kws
):
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


def test_mass(
    tend_mass, avg_dims_error=None, thresh=0.99999999, loc=None, iloc=None, plot=True, **plot_kws
):
    """Test closure of continuity equation.

    In the tendency calculations the vertical component of the continuity equation
    is calculated as residual to improve the budget closure which leads to automatic
    closure of the continuity equation.
    This test ensures that this residual calculation does not produce larger changes
    in the vertical component by comparing the residual calculation with the
    explicit calculation which uses the vertical velocity.
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
    fname = ""
    if "fname" in plot_kws:
        fname = plot_kws.pop("fname")
    for budget_form in dat.budget_form.values:
        dat_i = dat.sel(budget_form=budget_form)
        ref_i = ref.sel(budget_form=budget_form)
        dat_i = tools.loc_data(dat_i, loc=loc, iloc=iloc)
        ref_i = tools.loc_data(ref_i, loc=loc, iloc=iloc)
        e = R2(dat_i, ref_i, dim=avg_dims_error).min().values
        err.append(e)
        if e < thresh:
            log = "test_mass: vertical component of continuity equation\n for budget_form={}: min. R2 less than {}: {:.10f}".format(
                budget_form, thresh, e
            )
            print(log)
            if plot:
                dat_i.name = "Residual calculation"
                ref_i.name = "Calculation with vertical velocity"

                fname_i = fname
                if fname is not None:
                    fname_i = "budget_form=" + budget_form + "_" + fname
                    log = fname_i + "\n" + log
                plotting.scatter_hue(dat_i, ref_i, title=log, fname=fname_i, **plot_kws)
            failed = True
    return failed, min(err)


def test_adv_form(
    dat_mean,
    datout,
    var,
    cyclic=None,
    hor_avg=False,
    avg_dims=None,
    avg_dims_error=None,
    thresh=0.999,
    loc=None,
    iloc=None,
    plot=True,
    **plot_kws,
):
    """Compare implicit and explicit advective form calculations

    Explicitly calculate 2nd order mean advection in advective form and compare with
    implicit calculation.

    Parameters
    ----------
    dat_mean : xarray Dataset
        WRF time-averaged output.
    datout : dict
        Postprocessed output for variable var.
    var : str
        Variable to process.
    cyclic : dict of booleans for xy or None, optional
        Defines which dimensions have periodic boundary conditions.
        Use periodic boundary conditions to fill lateral boundary points.
        The default is None.
    hor_avg : bool, optional
        Horizontal averaging was used in postprocessing. The default is False.
    avg_dims : str or list of str, optional
        Averaging dimensions if hor_avg=True. The default is None.
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

    adv, flux, grid = datout["tend"]["adv"], datout["flux"], datout["grid"]
    dat_mean["bottom_top"] = flux["bottom_top"]
    dat_mean["bottom_top_stag"] = flux["bottom_top_stag"]
    vmean = xr.Dataset({"X": dat_mean["U_MEAN"], "Y": dat_mean["V_MEAN"], "Z": dat_mean["WD_MEAN"]})
    if var == "w":
        v = "ZWIND"
    else:
        v = var.upper()
    var_mean = dat_mean[v + "_MEAN"]
    if hor_avg:
        var_mean = tools.avg_xy(var_mean, avg_dims, cyclic=cyclic)

    vmean_c = xr.Dataset()
    dd = xr.Dataset()
    grad = xr.Dataset()
    tend = xr.Dataset()
    for dim in tools.XYZ:
        if hor_avg:
            vmean[dim] = tools.avg_xy(
                vmean[dim], avg_dims, cyclic=cyclic, **grid[tools.stagger_const]
            )
        ds = dim.lower()
        if dim == "Z":
            ds = "bottom_top"
        cyc = cyclic[ds]
        d = ds
        if ds in var_mean.dims:
            ds = ds + "_stag"
        else:
            d = d + "_stag"
        if dim == "Z":
            dd[dim] = tools.diff(grid["Z_STAG"], d, new_coord=flux[ds], cyclic=cyc)
        else:
            dd[dim] = grid["D" + dim]

        if d in adv.dims:
            grad[dim] = tools.diff(var_mean, d, new_coord=flux[ds], cyclic=cyc) / dd[dim]
            vmean_c[dim] = tools.stagger_like(
                vmean[dim], ref=grad[dim], cyclic=cyclic, **grid[tools.stagger_const]
            )

    for dim in tools.XYZ:
        if dim in grad:
            adv_s = -vmean_c[dim] * grad[dim]
            tend[dim] = tools.stagger_like(
                adv_s, ref=adv, cyclic=cyclic, **grid[tools.stagger_const]
            )
    for dim in ["X", "Y"]:
        if dim in grad:
            corr = grid[f"dzdt_{dim.lower()}"]
            corr = grad["Z"] * tools.stagger_like(
                corr, ref=grad["Z"], cyclic=cyclic, **grid[tools.stagger_const]
            )
            corr = tools.stagger_like(corr, ref=adv, cyclic=cyclic, **grid[tools.stagger_const])
            tend[dim] = tend[dim] - corr

    tend = tend.to_array("dir")

    fname = None
    if "fname" in plot_kws:
        fname = plot_kws.pop("fname")

    dat = tools.loc_data(
        adv.sel(budget_form="cartesian adv_form", dir=["X", "Y", "Z"], comp="mean"),
        loc=loc,
        iloc=iloc,
    )
    ref = tools.loc_data(tend, loc=loc, iloc=iloc)
    dat = dat.sel(dir=ref.dir)
    if var == "w":
        dat = dat.isel(bottom_top_stag=slice(1, None))
        ref = ref.isel(bottom_top_stag=slice(1, None))
    err = R2(dat, ref, dim=avg_dims_error).min().values
    failed = False
    if err < thresh:
        failed = True
        log = "test_adv_form: mean advective component: min. R2 less than {}: {:.10f}".format(
            thresh, err
        )
        print(log)
        if plot:
            dat.name = "Implicit calculation"
            ref.name = "Explicit calculation"
            if fname is not None:
                log = fname + "\n" + log
            plotting.scatter_hue(dat, ref, title=log, fname=fname, **plot_kws)
    return failed, err


def test_nan(datout, cut_bounds=None):
    """Test for NaN and inf values in postprocessed data.
    If invalid values occur, print reduced dataset to show their locations.

    Parameters
    ----------
    datout : dict
        Postprocessed output for variable var.
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


def test_y0(adv, thresh=(5e-6, 6e-3)):
    """Test whether the advective tendency resulting from fluxes in y-direction is,
    on average, much smaller than the one resulting from fluxes in x-direction. This
    should be the case if the budget is averaged over y. The average absolute ratio is
    compared to the given thresholds for the two budget forms "native" and "cartesian".
    """
    failed = False
    dims = [d for d in adv.dims if d not in ["dir", "budget_form", "comp"]]
    f = abs((adv.sel(dir="Y") / adv.sel(dir="X"))).median(dims)
    for budget_form, thresh_i in zip(["native", "cartesian"], thresh):
        if budget_form in f.budget_form:
            for comp in f.comp.values:
                fi = f.sel(budget_form=budget_form, comp=comp).values
                if fi > thresh_i:
                    print(
                        "test_y0 failed for budget_form={}, comp={}!: median(|adv_y/adv_x|) = {} > {}".format(
                            budget_form, comp, fi, thresh_i
                        )
                    )
                    failed = True
    return failed, f.max("comp").values


def test_dim_coords(dat, dat_inst, variable, dat_name, failed):
    """
    Test if dimension coordinates in postprocessed output are the same as in instantaneous WRF output.
    Exclude Time.

    """
    for dim in dat.dims:
        if (dim not in dat_inst.dims) or (dim == "Time"):
            continue
        c = dat[dim].values
        cr = dat_inst[dim].values
        if (len(c) != len(cr)) or (c != cr).any():
            print(
                "Coordinates for dimension {} in data {} of variable {}"
                " differs between postprocessed output and WRF output:"
                "\n {} vs. {}".format(dim, dat_name, variable, c, cr)
            )
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
        if (dat_inst["org"].SF_SURFACE_PHYSICS == 0) and (v == "LH"):
            # LH is arbitrary if no surface physics scheme is present
            continue
        try:
            if ("_open_" in ID) or ("_symm_" in ID):
                # with open and symmetric BC the results are not identical for some reason
                # TODO: why is this necessary?
                xr.testing.assert_allclose(dat_inst["debug"][v], dat_inst["org"][v], rtol=1e-6)
            else:
                xr.testing.assert_identical(dat_inst["debug"][v], dat_inst["org"][v])

        except AssertionError:
            print("Simulation with WRFlux and original WRF differ in variable {}!".format(v))
            res = "FAIL"
    return res


def test_periodic(datout, attrs, thresh=0.99999999, **kw):
    """Test if periodic boundary conditions are met.

    If boundary conditions are periodic, check if this is really the case
    in all postprocessed datasets: For the staggered dimensions x_stag and y_stag,
    test whether the left boundary values are equal to the right boundary values.

    Parameters
    ----------
    datout : dict
        Postprocessed output.
    attrs : dict
        Model settings
    thresh : float, optional
        Threshold value for R2 below which the test fails
    **kw :
        other keyword arguments (not used).

    Returns
    -------
    failed : bool
        Test failed.
    """
    failed = False
    for k, ds in datout.items():
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset()
        for dim in ["x", "y"]:
            dim_s = dim + "_stag"
            if attrs["PERIODIC_{}".format(dim.upper())]:
                for v in ds.variables:
                    if v == dim_s:
                        continue
                    if dim_s in ds[v].dims:
                        ref = ds[v][{dim_s: 0}]
                        dat = ds[v][{dim_s: -1}]
                        e = R2(dat, ref, dim=["x", "y", "bottom_top", "Time"]).min().values
                        if e < thresh:
                            log = (
                                f"test_periodic: {dim}-bounds not periodic for "
                                f" {v} in {k}: min. R2 less than {thresh}: {e:.10f}"
                            )
                            print(log)
                            failed = True
    return failed


# %% run_tests


def run_tests(
    datout,
    tests,
    dat_mean=None,
    dat_inst=None,
    sim_id="",
    trb_exp=False,
    hor_avg=False,
    chunks=None,
    figloc=None,
    **kw,
):
    """Run test functions for WRF output postprocessed with WRFlux.
       Thresholds are hard-coded.

    Parameters
    ----------
    datout : nested dict
        Postprocessed output for all variables.
    tests : list of str
        Tests to perform.
        Choices: testing.all_tests
    dat_mean : xarray Dataset
        WRF time-averaged output.
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
    figloc : str or path-like, optional
        Directory to save plot in. Defaults to the parent directory of this script.
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
    else:
        # drop duplicates
        tests = list(set(tests))
    tests = tests.copy()
    for test in tests:
        if test not in all_tests:
            raise ValueError(
                "Test {} not available! Available tests:\n{}".format(test, ", ".join(all_tests))
            )
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
    cyclic = {d: bool(attrs["PERIODIC_{}".format(d.upper())]) for d in tools.xy}
    cyclic["bottom_top"] = False

    avg_dims = None
    if hor_avg:
        avg_dims = []
        dat = datout[variables[0]]["tend"]["adv"]
        for d in tools.xy:
            if (d not in dat.dims) and (d + "_stag" not in dat.dims):
                avg_dims.append(d)

    # for w test: cut first time step
    if dat_inst is not None:
        dat_inst_lim = dat_inst.isel(Time=slice(1, None), **iloc)
    elif ("w" in tests) or ("dim_coords" in tests):
        raise ValueError("For tests 'w' and 'dim_coords', dat_inst needs to be given!")
    datout_lim = {}
    for v, datout_v in datout.items():
        datout_lim[v] = {}
        for n, dat in datout_v.items():
            if "budget_form" in dat.dims:
                budget_forms = []
                for budget_form in dat.budget_form.values:
                    budget_form = budget_form.split(" ")
                    budget_forms.append(" ".join(budget_form))
                dat["budget_form"] = budget_forms
            if "dim_coords" in tests:
                test_dim_coords(dat, dat_inst, v, n, failed)
            if hor_avg:
                for avg_dim in avg_dims:
                    for stag in ["", "_stag"]:
                        assert avg_dim + stag not in dat.dims
            datout_lim[v][n] = tools.loc_data(dat, iloc=iloc)

    if figloc is None:
        fpath = Path(__file__).parent
    else:
        fpath = Path(figloc)
    for var, datout_v in datout_lim.items():
        print("Variable: " + var)
        figloc = fpath / "figures" / var
        failed_i = {}
        err_i = {}
        if dat_mean is not None:
            dat_mean_v = dat_mean.sel(Time=datout_v["tend"]["Time"])

        if "budget" in tests:
            tend = datout_v["tend"]["net"].sel(side="tendency")
            forcing = datout_v["tend"]["net"].sel(side="forcing")
            kw["figloc"] = figloc / "budget"
            if (var == "w") and ("open BC y hor_avg" in sim_id):
                kw["thresh"] = 0.995
            elif (var in ["u", "v", "w"]) and ("open BC" in sim_id):
                kw["thresh"] = 0.998
            elif var == "t":
                if "open BC" in sim_id:
                    kw["thresh"] = 0.999
                if "symmetric BC" in sim_id:
                    kw["thresh"] = 0.995
                elif attrs["USE_THETA_M"] == 1:

                    if attrs["OUTPUT_DRY_THETA_FLUXES"] == 0:
                        # lower thresh as cartesian tendency for thm is close to 0
                        if attrs["MP_PHYSICS"] > 0:
                            kw["thresh_cartesian"] = 0.96
                            kw["thresh"] = 0.9998
                        else:
                            kw["thresh_cartesian"] = 0.995

                    # reduce threshold for WENO and monotonic advection as
                    # dry theta budget is not perfectly closed
                    elif (attrs["SCALAR_ADV_OPT"] >= 3) or (attrs["MOIST_ADV_OPT"] >= 3):
                        kw["thresh"] = 0.84
                    elif attrs["MOIST_ADV_OPT"] == 2:
                        kw["thresh"] = 0.96

            failed_i["budget"], err_i["budget"] = test_budget(tend, forcing, **kw)
            for thresh in ["thresh", "thresh_cartesian"]:
                if thresh in kw:
                    del kw[thresh]
        adv = datout_v["tend"]["adv"]
        if "decomp_sumdir" in tests:
            if attrs["HESSELBERG_AVG"] == 0:
                kw["thresh"] = 0.995
            elif trb_exp:
                kw["thresh"] = 0.998
            kw["figloc"] = figloc / "decomp_sumdir"
            failed_i["decomp_sumdir"], err_i["decomp_sumdir"] = test_decomp_sumdir(
                adv, datout_v["corr"], **kw
            )
            if "thresh" in kw:
                del kw["thresh"]

        if "decomp_sumcomp" in tests:
            if trb_exp:
                # reduce threshold for explicit turbulent fluxes
                kw["thresh"] = 0.998
            kw["figloc"] = figloc / "decomp_sumcomp"
            failed_i["decomp_sumcomp"], err_i["decomp_sumcomp"] = test_decomp_sumcomp(adv, **kw)
            if "thresh" in kw:
                del kw["thresh"]

        if ("w" in tests) and (var == variables[-1]):
            # only do test once: for last variable
            kw["figloc"] = figloc / "w"
            failed_i["w"], err_i["w"] = test_w(dat_inst_lim, **kw)

        if ("mass" in tests) and (var == "t"):
            if attrs["HESSELBERG_AVG"] == 0:
                kw["thresh"] = 0.99998

            kw["figloc"] = figloc / "mass"
            failed_i["mass"], err_i["mass"] = test_mass(datout_v["tend_mass"], **kw)
            if "thresh" in kw:
                del kw["thresh"]

        if "adv_form" in tests:
            kw["figloc"] = figloc / "adv_form"
            if var in ["u", "w"]:
                kw["thresh"] = 0.995
            if dat_mean is None:
                raise ValueError("For adv_form test, dat_mean needs to be given!")
            failed_i["adv_form"], err_i["adv_form"] = test_adv_form(
                dat_mean_v, datout_v, var, cyclic, hor_avg=hor_avg, avg_dims=avg_dims, **kw
            )
            if "thresh" in kw:
                del kw["thresh"]
        if "periodic" in tests:
            kw["figloc"] = figloc / "mass"
            failed_i["periodic"] = test_periodic(datout_v, attrs, **kw)
        if "NaN" in tests:
            failed_i["NaN"] = test_nan(datout_v)

        if "sgs" in tests:
            sgs_sum = datout_v["tend"]["adv"].sel(comp="trb_s").sum("dir")
            if np.allclose(sgs_sum[0], sgs_sum[1], atol=1e-7, rtol=1e-5):
                failed_i["sgs"] = False
            else:
                failed_i["sgs"] = True

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
    mse = ((dat - ref) ** 2).mean(**d)
    var = ((ref - ref.mean(**d)) ** 2).mean(**d)
    return 1 - mse / var


def trb_fluxes(
    dat_mean, inst, variables, grid, t_avg_interval, cyclic=None, hor_avg=False, avg_dims=None
):
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
    avg_kwargs = {
        "Time": t_avg_interval,
        "coord_func": {"Time": partial(tools.select_ind, indeces=-1)},
        "boundary": "trim",
    }

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
                rho = tools.build_mu(
                    inst["MUT_MEAN"], grid, full_levels="bottom_top_stag" in inst[var_d].dims
                )
            else:
                rho = inst["RHOD_MEAN"]

            var_d_m = means[var_d]
            vel_m = means[vel]
            if hor_avg:
                var_d_m = tools.avg_xy(
                    var_d_m, avg_dims, rho=rho, cyclic=cyclic, **grid[tools.stagger_const]
                )
                vel_m = tools.avg_xy(
                    vel_m, avg_dims, rho=rho, cyclic=cyclic, **grid[tools.stagger_const]
                )

            # compute perturbations
            var_pert = inst[var_d] - var_d_m
            rho_stag_vel = tools.stagger_like(
                rho, inst[vel], cyclic=cyclic, **grid[tools.stagger_const]
            )
            vel_pert = tools.stagger_like(
                rho_stag_vel * (inst[vel] - vel_m),
                var_pert,
                cyclic=cyclic,
                **grid[tools.stagger_const],
            )
            # build flux
            flux = vel_pert * var_pert
            flux = flux.coarsen(**avg_kwargs).mean()
            if hor_avg and (d.lower() not in avg_dims):
                flux = tools.avg_xy(flux, avg_dims, cyclic=cyclic)
                rho = tools.avg_xy(rho, avg_dims, cyclic=cyclic, **grid[tools.stagger_const])

            rho_stag = tools.stagger_like(rho, var_pert, cyclic=cyclic, **grid[tools.stagger_const])
            rho_stag_mean = rho_stag.coarsen(**avg_kwargs).mean()
            flux = flux / rho_stag_mean
            dat_mean["F{}{}_TRB_MEAN".format(var, v)] = flux
