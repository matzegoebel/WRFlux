#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:54:43 2020

Automatic tests for WRFlux.
Run WRF simulations with many different namelist settings, calculate tendencies, and
perform tests defined in testing.py.

@author: Matthias Göbel
"""
from run_wrf.launch_jobs import launch_jobs
from run_wrf.tools import grid_combinations, Capturing
import shutil
from collections import OrderedDict as odict
import os
from wrflux import tools
from wrflux.test import testing
import pandas as pd
import importlib
import numpy as np
import datetime
from pathlib import Path
import glob
from config import config_test_tendencies_base as conf
pd.set_option("display.precision", 15)
now = datetime.datetime.now().isoformat()[:16]
test_path = Path(__file__).parent.absolute()

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
skip_exist = True
# no running of simulations only postprocessing
skip_running = False

save_results = True  # save test results to csv tables

# builds to run simulations with
builds = ["org", "debug", "normal"]
# Change mapscale factors from 1 to random values around 1 to mimic real-case runs:
random_msf = True

# tests to perform
tests = testing.all_tests
# tests = ["budget", "decomp_sumdir", "decomp_sumcomp", "sgs",
#          "w", "mass", "Y=0", "NaN", "dim_coords",
#          "no_model_change", "periodic", "adv_form"]
# keyword arguments for tests (mainly for plotting)
kw = dict(
    avg_dims_error=["y", "bottom_top", "Time"],  # dimensions over which to calculate error norms
    # iloc={"x" : slice(5,-5)}, # integer-location based indexing before runnning tests
    # loc={"comp" : ["trb_r"]}, # label based indexing before runnning tests
    plot=True,
    # plot_diff=True, #plot difference between forcing and tendency against tendency
    discrete=True,  # discrete colormap
    # hue="comp",
    ignore_missing_hue=True,
    savefig=True,
)


# %% budget calculation methods
budget_methods = ["", "cartesian"]
budget_methods_advform = ["cartesian adv_form"]

# %%test functions

def test_all():
    """Define test simulations and start tests."""
    # Define parameter grid for simulations
    param_grids = {}
    th = {"use_theta_m": [0, 1, 1], "output_dry_theta_fluxes": [False, False, True]}
    o = np.arange(2, 7)

    # names of parameter values for output filenames
    # either dictionaries or lists (not for composite parameters)
    param_names = {"th": ["thd", "thm", "thdm"],
                   "output_streams": ["changed"],
                   "ieva": ["1"],
                   "h_adv_order": [2, 3],
                   "v_adv_order": [2, 3],
                   "adv_order": o,
                   "bc": ["open"],
                   "timing": ["short"],
                   "open_x": [True],
                   "open_y": [True],
                   "symm_x": [True],
                   "symm_y": [True]}

    # test processing only one variable at the time
    s = "output_{}_fluxes"
    d = {s.format(v): [2, 3] for v in tools.all_variables}
    param_names.update(d)
    d = {s.format(v) + "_1": [1] for v in tools.all_variables}
    param_names.update(d)
    for v in tools.all_variables:
        sv = s.format(v)
        d = {s.format(vi): [0,0] for vi in tools.all_variables}
        d[sv] = [2, 3]
        param_grids[sv + "_debug_only"] = {sv: d}
        d = {s.format(vi): [0] for vi in tools.all_variables}
        d[sv] = [1]
        param_grids[sv + "_no_org"] = {sv + "_1": d}
    param_grids["trb no_debug"] = odict(msf=1, input_sounding="wrflux_u",
        timing=dict(
        end_time=["2018-06-20_12:30:00"],
        output_streams=[{24: ["meanout", conf.params["dt_f"] / 60.],
                          0: ["instout", 10.]}]))
    param_grids["trb no_debug hor_avg"] = param_grids["trb no_debug"].copy()
    param_grids["hor_avg no_debug msf=1"] = odict(msf=1, input_sounding="wrflux_u")  # for Y=0 test
    param_grids["adv_form"] = odict(msf=1, input_sounding="wrflux_u",
                                    adv_order=dict(h_sca_adv_order=[2], v_sca_adv_order=[2],
                                                   h_mom_adv_order=[2], v_mom_adv_order=[2]))
    param_grids["adv_form hor_avg no_debug"] = param_grids["adv_form"].copy()
    param_grids["hor_avg no_debug"] = odict()
    param_grids["chunking xy no_debug"] = odict(chunks={"x": 10, "y": 10})
    param_grids["chunking x no_debug"] = odict(chunks={"x": 10})
    param_grids["chunking x hor_avg no_debug"] = param_grids["chunking x no_debug"].copy()
    param_grids["no density-weighting"] = odict(hesselberg_avg=[False])  # TODO also skip hessel in postproc?
    param_grids["serial"] = odict(lx=[5000], ly=[5000])
    param_grids["avg_interval"] = odict(avg_interval=1200, output_streams=[{24: ["meanout", 30.], 0: ["instout", 10.]}])
    param_grids["km_opt"] = odict(km_opt=[2, 5], spec_hfx=[0.2, None], th=th)
    param_grids["PBL scheme with theta moist+dry"] = odict(bl_pbl_physics=[1], th=th)
    param_grids["2nd-order advection th variations"] = odict(use_theta_m=[0, 1],
                                                             adv_order=dict(h_sca_adv_order=2,
                                                                            v_sca_adv_order=2,
                                                                            h_mom_adv_order=2,
                                                                            v_mom_adv_order=2))
    param_grids["simple and positive-definite advection"] = odict(
        moist_adv_opt=[0, 1],
        adv_order=dict(h_sca_adv_order=o, v_sca_adv_order=o, h_mom_adv_order=o, v_mom_adv_order=o))
    dx = 1000
    param_grids["ieva"] = odict(use_theta_m=1, ieva=dict(zadvect_implicit=1, dt_f=10, dx=dx,
                                lx=20*dx, ly=20*dx, dzmax=100, nz=None, hm=0, spec_hfx=0.3))

    param_grids["WENO advection"] = odict(
        moist_adv_opt=[3, 4], scalar_adv_opt=[3], momentum_adv_opt=[3], th=th)
    param_grids["monotonic advection"] = odict(moist_adv_opt=[2], v_sca_adv_order=[3, 5], th=th)
    param_grids["MP + CU"] = odict(cu_physics=16, shcu_physics=2, bl_pbl_physics=9, mp_physics=2, th=th)
    param_grids["damp2_diff6"] = odict(damp_opt=2, diff_6th_opt=1, th=th)
    param_grids["damp3"] = odict(damp_opt=3)
    param_grids["w_damping"] = odict(w_damping=1, dz0=30, dzmax=50, dx=1000, lx=20000, ly=20000, dt_f=10)

    hm = 0  # flat simulations in boundaries are not periodic
    param_grids["open BC x"] = odict(open_x=dict(open_xs=[True], open_xe=[True], periodic_x=[False],
                                     hm=hm, spec_hfx=[0.2], input_sounding="free"))
    param_grids["open BC y"] = odict(open_y=dict(open_ys=[True], open_ye=[True], periodic_y=[False],
                                     hm=hm, spec_hfx=[0.2], input_sounding="free"))
    param_grids["open BC y hor_avg"] = param_grids["open BC y"].copy()
    param_grids["symmetric BC x"] = odict(symm_x=dict(symmetric_xs=[True], symmetric_xe=[True], periodic_x=[False],
                                          hm=hm, spec_hfx=[0.2], input_sounding="free"))
    param_grids["symmetric BC y"] = odict(symm_y=dict(symmetric_ys=[True], symmetric_ye=[True], periodic_y=[False],
                                          hm=hm, spec_hfx=[0.2], input_sounding="free"))
    param_grids["symmetric BC y hor_avg"] = param_grids["symmetric BC y"].copy()

    failed, failed_short, err, err_short, err_diff, err_ratio = run_and_test(param_grids, param_names, avg_dims=["y"])

    return failed, failed_short, err, err_short, err_diff, err_ratio


# %% run_and_test

def run_and_test(param_grids, param_names, avg_dims=None):
    """Run test simulations defined by param_grids and config_file and perform tests."""
    index = pd.MultiIndex.from_product([["INIT", "RUN"] + tests, variables])
    failed = pd.DataFrame(index=index)
    failed_short = pd.DataFrame(columns=variables)
    index = pd.MultiIndex.from_product([tests, variables])
    err = pd.DataFrame(index=index)

    test_results = test_path / "test_results"
    os.makedirs(test_results, exist_ok=True)

    for label, param_grid in param_grids.items():
        print("\n\n\n{0}\n{0}\n{0}\nRun test simulations: {1}\n{0}\n{0}\n{0}\n".format("#" * 70, label))
        # initialize and run simulations

        builds_i = builds.copy()
        variables_i = variables
        for v in variables:
            if f"output_{v}_fluxes" in label:
                variables_i = [v]
                break
        runID = None
        if "no_debug" in label:
            builds_i = [b for b in builds if b not in ["debug", "org"]]
        elif "debug_only" in label:
            builds_i = ["debug"]
        elif "no_org" in label:
            builds_i = ["normal", "debug"]
        if "chunks" in param_grid:
            chunks = param_grid.pop("chunks")
        else:
            chunks = None
        rmsf = random_msf
        if ("msf" in param_grid.keys()) and (param_grid["msf"] == 1):
            rmsf = False
            del param_grid["msf"]
            runID = "pytest_msf=1"

        skip_exist_i = skip_exist
        if ((exist != "s") and (not skip_running)) or ("no_debug" in label):
            # if simulation is repeated or param_grid is empty (focus on different postprocessing options),
            # also repeat postprocessing
            skip_exist_i = False

        for build in tools.make_list(builds_i):
            print("\n\n\n{0}\n{0}\nbuild: {1}\n{0}\n{0}\n".format("#" * 50, build))

            conf, cfile = load_config(build)

            if runID is None:
                runID_i = conf.runID
            else:
                runID_i = runID
            runID_i += "_" + build

            param_grid_i = param_grid.copy()
            param_combs = grid_combinations(param_grid_i, conf.params, param_names=param_names,
                                            runID=runID_i)

            if not skip_running:
                # setup module_initialize_ideal.F
                build_dir = setup_test_sim(build, random_msf=rmsf)

                # delete parameters not available in original WRF
                if build == "org":
                    for p in ["output_dry_theta_fluxes", "hesselberg_avg", "avg_interval"]:
                        if p in param_combs:
                            del param_combs[p]

                # initialize simulations
                combs, output = capture_submit(init=True, exist=exist, build=build_dir, config_file=cfile,
                                               param_combs=param_combs)
                # run simulations
                combs, output = capture_submit(init=False, wait=True, build=build_dir, pool_jobs=True,
                                               exist=exist, config_file=cfile, param_combs=param_combs)
                print("\n\n")

            # check if all simulations were successful
            for cname, param_comb in param_combs.iterrows():
                if cname in ["core_param", "composite_idx"]:
                    continue
                IDi = param_comb["fname"]
                ind = label + ": " + IDi
                print("\n\n{0}\nCheck simulation: {1}, {2}\n{0}\n".format("#" * 70, label, IDi))
                failed[ind] = ""
                err[ind] = ""
                print("Check if simulations were successfully initialized and run.")
                run_dir = Path(conf.params["run_path"]) / ("WRF_" + IDi + "_0")
                log = (run_dir / "init.log").read_text()
                if "wrf: SUCCESS COMPLETE IDEAL INIT" not in log:
                    print("Error in initializing simulation!")
                    failed[ind].loc["INIT", variables_i[0]] = "FAIL"
                    continue
                log = (run_dir / "rsl.error.0000").read_text()
                if "wrf: SUCCESS COMPLETE WRF" not in log:
                    print("Error in running simulation!")
                    failed[ind].loc["RUN", variables_i[0]] = "FAIL"
                    continue

                # postprocess data and run tests
                outpath_c = os.path.join(conf.params["outpath"], IDi) + "_0"
                start_time = param_comb["start_time"]
                inst_file = "instout_d01_" + start_time
                mean_file = "meanout_d01_" + start_time

                if "normal" in builds_i:
                    use_build = "normal"
                else:
                    use_build = "debug"
                if ("no_model_change" in tests) and ("debug" in builds_i) and \
                   ("org" in builds_i) and (build == use_build):
                    print("Check for differences between debug build and official WRF.")
                    # replace build with placeholder for later formatting
                    ID_b = IDi.replace(build, "{}")
                    f = testing.test_no_model_change(conf.params["outpath"], ID_b, inst_file, mean_file)
                    failed.loc["no_model_change", variables_i[0]] = f

                if build != "normal":
                    continue

                print("Postprocess simulation")

                # set budget methods
                bm = budget_methods
                tests_i = tests.copy()
                hor_avg = False
                if "hor_avg" in label:
                    hor_avg = True
                if ("chunking" in label) and ("periodic" in tests_i):
                    tests_i.remove("periodic")
                if "adv_form" in label:
                    bm = budget_methods + budget_methods_advform
                elif "adv_form" in tests_i:
                    tests_i.remove("adv_form")

                t_avg = False
                t_avg_interval = None
                if "trb " in label:
                    if "w" in tests_i:
                        tests_i.remove("w")
                    t_avg = True
                    t_avg_interval = int(param_comb["output_streams"][0][1] * 60 / param_comb["dt_f"])

                datout, dat_inst, dat_mean = tools.calc_tendencies(
                    variables_i, outpath_c, inst_file=inst_file, mean_file=mean_file, budget_methods=bm,
                    hor_avg=hor_avg, avg_dims=avg_dims, t_avg=t_avg, t_avg_interval=t_avg_interval,
                    skip_exist=skip_exist_i, save_output=True, return_model_output=True, chunks=chunks)

                print("\n{}\nRun tests:\n".format("#" * 10))
                if rmsf and ("Y=0" in tests_i):
                    tests_i.remove("Y=0")
                kw["fname"] = label.replace(" ", "_") + ":" + IDi  # figure filename
                kw["close"] = True  # always close figures
                failed_i, err_i = testing.run_tests(datout, tests_i, dat_mean=dat_mean, dat_inst=dat_inst,
                                                    sim_id=ind, hor_avg=hor_avg, trb_exp=t_avg,
                                                    chunks=chunks, **kw)

                for var in variables_i:
                    for test, f in failed_i.loc[var].items():
                        failed[ind].loc[test, var] = f
                    for test, e in err_i.loc[var].items():
                        err[ind].loc[test, var] = e
                if save_results:
                    failed.to_csv(test_results / ("test_results_" + now + ".csv"))
                    err.to_csv(test_results / ("test_scores_" + now + ".csv"))

    if restore_init_module and (not skip_running):
        print("\n")
        for build in tools.make_list(builds):
            print("\nRestore files for build " + build)
            setup_test_sim(build, restore=True)

    # assemble and save final test results
    for ind in failed.keys():
        for var in variables:
            failed_v = failed[ind].loc[:, var]
            failed_tests = ",".join([test for test, f in failed_v.iteritems() if f == "FAIL"])
            failed_short.loc[ind, var] = failed_tests
    failed_short = failed_short.where(failed_short.astype(str) != "").dropna(how="all").dropna(axis=1, how="all")
    failed_short = failed_short.where(~failed_short.isnull(), "")
    err_short = err.where(failed == "FAIL")
    err_short = err_short.where(~err_short.isnull(), "")
    err_short = err_short.where(err_short.astype(str) != "").dropna(how="all").dropna(axis=1, how="all")
    err_short = err_short.where(~err_short.isnull(), "")
    err = err.where(err.astype(str) != "").dropna(how="all").dropna(axis=1, how="all")
    err = err.where(~err.isnull(), "")
    failed = failed.where(failed.astype(str) != "").dropna(how="all").dropna(axis=1, how="all")
    failed = failed.where(~failed.isnull(), "")

    # load previous scores file and compute difference
    err_previous = glob.glob(str(test_results / "test_scores_*.csv"))
    err_previous = sorted([f for f in err_previous if "failsonly" not in f and now not in f])
    err_diff = None
    err_ratio = None
    if len(err_previous) > 0:
        err_previous = pd.read_csv(err_previous[-1], header=0, index_col=(0, 1))
        # delete Y=0 test
        err_previous = err_previous.loc[[t for t in err_previous.index if t[0] != "Y=0"]].astype(float)
        err_clean = err.where(err.astype(str) != "")
        err_clean = err_clean.loc[[t for t in err_clean.index if t[0] != "Y=0"]]
        err_clean = err_clean.astype(float)
        err_diff = err_clean - err_previous
        err_ratio = (1 - err_clean) / (1 - err_previous)
        err_diff = err_diff.dropna(0, "all").dropna(1, "all")
        err_ratio = err_ratio.dropna(0, "all").dropna(1, "all")

    if save_results:
        failed.to_csv(test_results / ("test_results_" + now + ".csv"))
        err.to_csv(test_results / ("test_scores_" + now + ".csv"))
        failed_short.to_csv(test_results / ("test_results_failsonly_" + now + ".csv"))
        err_short.to_csv(test_results / ("test_scores_failsonly_" + now + ".csv"))

    if (failed_short.astype(str) != "").values.any():
        message = "\n\n{}\nFailed tests:\n{}".format("#" * 100, failed_short.to_string())
        print(message)
        if raise_error:
            raise RuntimeError(message)

    return failed, failed_short, err, err_short, err_diff, err_ratio


# %% misc


def capture_submit(*args, **kwargs):
    """Capture output of launch_jobs and print it only if an error occurs."""
    try:
        with Capturing() as output:
            combs = launch_jobs(*args, **kwargs)
    except Exception as e:
        print("\n".join(output))
        raise(e)

    return combs, output


def load_config(build):
    """Load config module for given build.

    Parameters
    ----------
    build : str
        WRF build: 'normal', 'debug', or 'org'.

    Returns
    -------
    conf : module
        Configuration module for test simulations.
    cfile : str
        Path to conf.
    """
    cfile = "wrflux.test.config.config_test_tendencies"
    conf = importlib.import_module(cfile)
    if build == "debug":
        cfile = cfile + "_debug"
    elif build == "org":
        cfile = cfile + "_base"
    conf = importlib.import_module(cfile)

    return conf, cfile


def setup_test_sim(build, restore=False, random_msf=True):
    """Setup test simulation.

    Replace module_initialize_ideal.F with test file and recompile.
    Copy input sounding and IO_file to case directory.

    Parameters
    ----------
    build : str
        WRF build: 'normal', 'debug', or 'org'.
    restore : bool, optional
        Restore original module_initialize_ideal.F after tests are finished and recompile.
        The default is True.
    random_msf : bool, optional
        Change mapscale factors from 1 to random values around 1 to mimic real-case runs.
        The default is True.

    Returns
    -------
    build_dir : str
        WRF build directory.
    """
    conf, cfile = load_config(build)
    if build == "debug":
        build_dir = conf.params["debug_build"]
    elif build == "org":
        build_dir = conf.org_build
    else:
        build_dir = conf.params["parallel_build"]

    #  setup module_initialize_ideal.F
    fname = "module_initialize_ideal.F"
    wrf_path = Path(conf.params["build_path"]) / build_dir
    fpath = wrf_path / "dyn_em" / fname
    recompile = True
    fpath_org = Path(str(fpath) + ".org")
    input_path = test_path / "input"
    if restore:
        m = "Restore"
        if os.path.isfile(fpath_org):
            os.rename(fpath_org, fpath)
    else:
        org_file = fpath.read_text()
        testf = "TEST_"
        if random_msf:
            testf += "msf_"
        test_file_path = (input_path / (testf + fname))
        test_file = test_file_path.read_text()
        if test_file == org_file:
            recompile = False
        else:
            m = "Copy"
            if not fpath_org.exists():
                shutil.copy(fpath, fpath_org)
            shutil.copy(test_file_path, fpath)
    if recompile:
        print(m + " module_initialize_ideal.F and recompile")
        os.chdir(wrf_path)
        err = os.system(str(wrf_path / "compile") + " em_les > log 2> err")
        if err != 0:
            raise RuntimeError("WRF compilation failed!")
        os.chdir(test_path)

    case_path = wrf_path / "test" / conf.params["ideal_case_name"]

    # IO file, input sounding, and namelist file
    input_files = ["IO_wdiag.txt", "namelist.input"]
    for f in os.listdir(input_path):
        if "input_sounding" in f:
            input_files.append(f)

    for f in input_files:
        fpath = case_path / f
        fpath_org = Path(str(fpath) + ".org")
        if not restore:
            src = f
            if f == "namelist.input":
                if build == "org":
                    src = src + ".org"
                else:
                    src = src + ".wrflux"
            src = input_path / src
            if (not fpath.exists()) or (fpath.read_text() != src.read_text()):
                print("Copy {}".format(f))
                if f == "namelist.input":
                    if not fpath_org.exists():
                        shutil.copy(fpath, fpath_org)
                shutil.copy(src, fpath)
        else:
            if os.path.isfile(fpath_org):
                os.remove(fpath)
                if f == "namelist.input":
                    print("Restore namelist file")
                    os.rename(fpath_org, fpath)

    return build_dir


# %%main
if __name__ == "__main__":
    failed, failed_short, err, err_short, err_diff, err_ratio = test_all()

    err_dict = {}
    err_short_dict = {}
    for e_df, e_dict in zip([err, err_short], [err_dict, err_short_dict]):
        for test in e_df.index.levels[0]:
            if test in e_df.index:
                e = e_df.loc[test]
                e = e.where(e.astype(str) != "").dropna(how="all").dropna(axis=1, how="all")
                e_dict[test] = e.where(~e.isnull(), "").T
