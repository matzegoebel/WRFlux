#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:54:43 2020

Automatic tests for WRFlux.
Run WRF simulations with many different namelist settings, calculate tendencies, and
perform tests defined in testing.py.

@author: Matthias Göbel
"""
import sys
from run_wrf.launch_jobs import launch_jobs
from run_wrf.tools import grid_combinations, Capturing
import shutil
from collections import OrderedDict as odict
import os
from wrflux import tools
from wrflux.test import testing
import pandas as pd
import pytest
import importlib
import numpy as np
XY = ["X", "Y"]

# %% settings

variables = ["q", "t", "u", "v", "w"]

# raise if tests fail
raise_error = True
# restore module_initialize_ideal.F after running tests
restore_init_module = True
# What to do if simulation output already exists: Skip run ('s'), overwrite ('o'),
# restart ('r') or backup files ('b').
exist = "s"
# skip postprocessing if data already exists
skip_exist = False
# run simulations with debug and/or normal built
debug = [True, False]
# Change mapscale factors from 1 to random values around 1 to mimic real-case runs:
random_msf = True
# tests to perform
tests = ["budget", "decomp_sumdir", "decomp_sumcomp", "dz_out", "adv_2nd", "w", "Y=0", "NaN"]
# Mapping from dimension "x" and/or "y" to chunk sizes to split the domain in tiles
chunks = None
# chunks = {"x" : 10} #TODOm: problem with trb runs

# keyword arguments for tests (mainly for plotting)
kw = dict(
    avg_dims_error=["y", "bottom_top", "Time"],  # dimensions over which to calculate error norms
    plot=True,
    # plot_diff = True, #plot difference between forcing and tendency against tendency
    discrete=True,  # discrete colormap
    # iloc={"x" : slice(5,-5)},
    # hue="comp",
    ignore_missing_hue=True,
    savefig=True,
    # close = True  # close figures directly
)


# %% budget calculation methods
budget_methods = [
                 [],
                 ["cartesian", "correct"]]
budget_methods_2nd = [
                     ["cartesian", "2nd"],
                     ["cartesian", "correct", "2nd"],
                     ["cartesian"]]
budget_methods_dzout = [
                       ["cartesian", "correct", "dz_out"],
                       ["cartesian", "correct", "dz_out", "corr_varz"]]


# %%test functions

def test_all():
    """Define test simulations and start tests."""
    # Define parameter grid for simulations
    param_grids = {}
    th = {"use_theta_m": [0, 1, 1], "output_dry_theta_fluxes": [False, False, True]}
    th2 = {"use_theta_m": [0, 1], "output_dry_theta_fluxes": [False, False]}
    o = np.arange(2, 7)

    ### param_grids["2nd"] =  odict(adv_order=dict(h_sca_adv_order=[2], v_sca_adv_order=[2], h_mom_adv_order=[2], v_mom_adv_order=[2]))
    param_grids["dz_out msf=1"] = odict(runID="dz_out")
    param_grids["trb no_debug msf=1"] = odict(timing=dict(
        end_time=["2018-06-20_12:30:00"],
        output_streams=[{24: ["meanout", 2. / 60.], 0: ["instout", 10.]}]))
    param_grids["trb no_debug hor_avg msf=1"] = param_grids["trb no_debug msf=1"]
    param_grids["hor_avg msf=1"] = odict(runID="hor_avg_msf=1")  # for Y=0 test
    param_grids["hor_avg"] = odict(runID="hor_avg")
    param_grids["hessel"] = odict(hesselberg_avg=[False])
    param_grids["serial"] = odict(lx=[5000], ly=[5000])
    param_grids["km_opt"] = odict(km_opt=[2, 5], spec_hfx=[0.2, None], th=th)
    param_grids["no small fluxes"] = odict(th=th, output_t_fluxes_small=[0])
    param_grids["PBL scheme with theta moist/dry"] = odict(bl_pbl_physics=[1], th=th)
    param_grids["2nd-order advection th variations"] = odict(use_theta_m=[0, 1], h_sca_adv_order=2,
                                                             v_sca_adv_order=2, h_mom_adv_order=2,
                                                             v_mom_adv_order=2)
    param_grids["simple and positive-definite advection"] = odict(
        moist_adv_opt=[0, 1],
        adv_order=dict(h_sca_adv_order=o, v_sca_adv_order=o, h_mom_adv_order=o, v_mom_adv_order=o))
    param_grids["WENO advection"] = odict(
        moist_adv_opt=[0, 3, 4], scalar_adv_opt=[3], momentum_adv_opt=[3], th=th2)
    param_grids["monotonic advection"] = odict(moist_adv_opt=[2], v_sca_adv_order=[3, 5], th=th2)
    param_grids["MP rad"] = odict(mp_physics=[2], th=th)

    hm = 0
    param_grids["open BC x"] = odict(open_xs=[True], open_xe=[True], periodic_x=[False],
                                     hm=hm, spec_hfx=[0.2], input_sounding="free")
    param_grids["open BC y"] = odict(open_ys=[True], open_ye=[True], periodic_y=[False],
                                     hm=hm, spec_hfx=[0.2], input_sounding="free")
    param_grids["open BC y hor_avg"] = param_grids["open BC y"]
    param_grids["symmetric BC x"] = odict(symmetric_xs=[True], symmetric_xe=[True], periodic_x=[False],
                                          hm=hm, spec_hfx=[0.2], input_sounding="free")
    param_grids["symmetric BC y"] = odict(symmetric_ys=[True], symmetric_ye=[True], periodic_y=[False],
                                          hm=hm, spec_hfx=[0.2], input_sounding="free")
    param_grids["symmetric BC y hor_avg"] = param_grids["symmetric BC y"]

    failed, failed_short, err, err_short = run_and_test(param_grids, avg_dims=["y"])

    return failed, failed_short, err, err_short


# %% run_and_test

def run_and_test(param_grids, config_file="wrflux.test.config_test_tendencies", avg_dims=None):
    """Run test simulations defined by param_grids and config_file and perform tests."""
    index = pd.MultiIndex.from_product([["INIT", "RUN"] + tests, variables])
    failed = pd.DataFrame(index=index)
    failed_short = pd.DataFrame(columns=variables)
    index = pd.MultiIndex.from_product([tests, variables])
    err = pd.DataFrame(index=index)
    skip_exist_i = skip_exist
    if exist != "s":
        skip_exist_i = False

    for label, param_grid in param_grids.items():
        print("\n\n\n{0}\nRun test simulations: {1}\n{0}\n".format("#" * 50, label))
        # initialize and run simulations
        debugs = debug
        if "no_debug" in label:
            debugs = False
        if "runID" in param_grid:
            runID = param_grid.pop("runID")
        else:
            runID = None

        for deb in tools.make_list(debugs):
            cfile = config_file
            if deb:
                cfile = cfile + "_debug"
            conf = importlib.import_module(cfile)
            rmsf = random_msf
            if "msf=1" in label:
                rmsf = False
            if runID is None:
                runID_i = conf.runID
            elif deb:
                runID_i = runID + "_debug"
            else:
                runID_i = runID
            setup_test_init_module(conf, debug=deb, random_msf=rmsf)
            param_combs = grid_combinations(param_grid, conf.params, param_names=conf.param_names,
                                            runID=runID_i)
            combs, output = capture_submit(init=True, exist=exist, debug=deb, config_file=cfile,
                                           param_combs=param_combs)
            combs, output = capture_submit(init=False, wait=True, debug=deb, pool_jobs=True,
                                           exist=exist, config_file=cfile, param_combs=param_combs)
            print("\n\n")

            for cname, param_comb in param_combs.iterrows():
                if cname in ["core_param", "composite_idx"]:
                    continue
                IDi = param_comb["fname"]
                ind = label + ": " + IDi
                failed[ind] = ""
                # check logs
                run_dir = os.path.join(conf.params["run_path"], "WRF_" + IDi + "_0")
                with open("{}/init.log".format(run_dir)) as f:
                    log = f.read()
                if "wrf: SUCCESS COMPLETE IDEAL INIT" not in log:
                    print("Error in initializing simulation {}!".format(cname))
                    failed[ind].loc["INIT", variables[0]] = "FAIL"
                    continue
                with open("{}/run.log".format(run_dir)) as f:
                    log = f.read()
                if "wrf: SUCCESS COMPLETE WRF" not in log:
                    print("Error in running simulation {}!".format(cname))
                    failed[ind].loc["RUN", variables[0]] = "FAIL"
                    continue

        for cname, param_comb in param_combs.iterrows():
            if cname in ["core_param", "composite_idx"]:
                continue
            IDi = param_comb["fname"]
            ind = label + ": " + IDi
            if (failed.loc[["RUN", "INIT"], ind] == "FAIL").any():
                continue
            print("\n\n\n{0}\nPostprocess simulation: {1}, {2}\n{0}\n".format("#" * 50, label, cname))

            err[ind] = ""
            # postprocessing
            print("Postprocess data")
            bm = budget_methods
            hor_avg = False
            t_avg = False
            t_avg_interval = None
            tests_i = tests.copy()
            if "hor_avg" in label:
                hor_avg = True

            if "dz_out" in label:
                bm = bm + budget_methods_dzout
            elif "dz_out" in tests_i:
                tests_i.remove("dz_out")

            if all(param_comb[i + "_adv_order"] == 2 for i in ["h_sca", "v_sca", "h_mom", "v_mom"]):
                bm = bm + budget_methods_2nd
            elif "adv_2nd" in tests_i:
                tests_i.remove("adv_2nd")

            if "trb " in label:
                if "w" in tests_i:
                    tests_i.remove("w")
                t_avg = True
                t_avg_interval = int(param_comb["output_streams"][0][1] * 60 / param_comb["dt_f"])

            outpath_c = os.path.join(conf.params["outpath"], IDi) + "_0"
            datout, dat_inst, dat_mean = tools.calc_tendencies(
                variables, outpath_c, start_time=param_comb["start_time"], budget_methods=bm,
                hor_avg=hor_avg, avg_dims=avg_dims, t_avg=t_avg, t_avg_interval=t_avg_interval,
                skip_exist=skip_exist_i, save_output=True, return_model_output=True, chunks=chunks)

            print("\n\n\n{0}\nRun tests\n{0}\n".format("#" * 50))
            if rmsf and ("Y=0" in tests_i):
                tests_i.remove("Y=0")
            # figure filename
            kw["fname"] = label.replace(" ", "_") + ":" + IDi
            failed_i, err_i = testing.run_tests(datout, tests_i, dat_inst=dat_inst,
                                                hor_avg=hor_avg, trb_exp=t_avg, chunks=chunks, **kw)

            for var in variables:
                for test, f in failed_i.loc[var].items():
                    failed[ind].loc[test, var] = f
                for test, e in err_i.loc[var].items():
                    err[ind].loc[test, var] = e

    if restore_init_module:
        for deb in tools.make_list(debug):
            cfile = config_file
            if deb:
                cfile = cfile + "_debug"
            conf = importlib.import_module(cfile)
            setup_test_init_module(conf, debug=deb, restore=True)

    for ind in failed.keys():
        for var in variables:
            failed_tests = ",".join([test for test, f in failed[ind].loc[:, var].iteritems() if f == "FAIL"])
            failed_short.loc[ind, var] = failed_tests

    failed_short = failed_short.where(failed_short != "").dropna(how="all").dropna(axis=1, how="all")
    failed_short = failed_short.where(~failed_short.isnull(), "")

    err_short = err.where(failed == "FAIL").dropna(how="all").dropna(axis=1, how="all")
    err_short = err_short.where(~err_short.isnull(), "")
    err = err.where(err != "").dropna(how="all").dropna(axis=1, how="all")
    err = err.where(~err.isnull(), "")
    failed = failed.where(failed != "").dropna(how="all").dropna(axis=1, how="all")
    failed = failed.where(~failed.isnull(), "")
    if (failed_short != "").values.any():
        message = "\n\n{}\nFailed tests:\n{}".format("#" * 100, failed_short.to_string())
        if raise_error:
            raise RuntimeError(message)
        else:
            print(message)
    return failed, failed_short, err, err_short


# %% misc
@pytest.fixture(autouse=True)
def run_around_tests(request):
    """Delete test data before and after every test"""
    # Code that will run before each test
    delete_test_data()
    # A test function will be run at this point
    yield
    # Code that will run after each test
    delete_test_data()


def delete_test_data():
    """Delete run directories and results produced by this test suite."""
    for d in [os.environ["wrf_res"] + "/test/pytest", os.environ["wrf_runs"] + "/test/pytest"]:
        if os.path.isdir(d):
            shutil.rmtree(d)


def capture_submit(*args, **kwargs):
    """Capture output of launch_jobs and print it only if an error occurs."""
    try:
        with Capturing() as output:
            combs = launch_jobs(*args, **kwargs)
    except Exception as e:
        print("\n".join(output))
        raise(e)

    return combs, output


def setup_test_init_module(conf, debug=False, restore=False, random_msf=True):
    """Replace module_initialize_ideal.F with test file and recompile.

    Parameters
    ----------
    conf : module
        Configuration module for test simulations.
    debug : bool, optional
        Copy file to debug build instead of normal build. The default is False.
    restore : bool, optional
        Restore original module_initialize_ideal.F after tests are finished and recompile.
        The default is True.
    random_msf : bool, optional
        Change mapscale factors from 1 to random values around 1 to mimic real-case runs.
        The default is True.

    Returns
    -------
    None.

    """
    fname = "module_initialize_ideal.F"
    if debug:
        build = conf.debug_build
    else:
        build = conf.parallel_build
    wrf_path = "{}/{}".format(conf.params["build_path"], build)
    fpath = wrf_path + "/dyn_em/" + fname

    if restore:
        m = "Restore"
        shutil.copy(fpath + ".org", fpath)
    else:
        with open(fpath) as f:
            org_file = f.read()
        test_path = os.path.abspath(os.path.dirname(__file__))
        testf = "TEST_"
        if random_msf:
            testf += "msf_"
        with open(test_path + "/" + testf + fname) as f:
            test_file = f.read()
        if test_file == org_file:
            return

        # copy file and recompile
        m = "Copy"
        shutil.copy(fpath, fpath + ".org")
        shutil.copy(test_path + "/" + testf + fname, fpath)
        print(m + " module_initialize_ideal.F and recompile")
        os.chdir(wrf_path)
        os.system("./compile em_les > log 2> err")
        os.chdir(test_path)


# %%main
if __name__ == "__main__":
    failed, failed_short, err, err_short = test_all()
