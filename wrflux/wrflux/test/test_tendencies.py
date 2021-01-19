#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:54:43 2020

Calculate the WRF budget terms for theta or qv under different assumptions.
Scatter and profile plots of the individual terms and sums.

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
exist = "s"

variables = ["q", "t", "u", "v", "w"]
# variables = ["u"]

raise_error = False
restore_init_module = True
skip_exist = False
debug = [True, False]

random_msf = True  # change mapscale factors from 1 to random values
tests = ["budget", "decomp_sumdir", "decomp_sumcomp", "dz_out", "adv_2nd", "w", "Y=0", "NaN"]
chunks = None
# chunks = {"x" : 10} #TODOm: problem with trb runs
kw = dict(
    avg_dims_error=["y", "bottom_top", "Time"],  # dimensions over which to calculate error norms
    plot=True,
    # plot_diff = True, #plot difference between forcing and tendency against tendency
    discrete=True,
    # iloc={"x" : [*np.arange(9),*np.arange(-9,0)]},
    # iloc={"y" : [*np.arange(-9,0)]},
    # hue="comp",
    # s=40,
    ignore_missing_hue=True,
    # savefig = False,#TODOm
    # close = True
)
# %%set calculation methods
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

if exist != "s":
    skip_exist = False
# %%test functions


def test_all():
    # Define parameter grid for simulations
    param_grids = {}
    th = {"use_theta_m": [0, 1, 1], "output_dry_theta_fluxes": [False, False, True]}
    th2 = {"use_theta_m": [0, 1], "output_dry_theta_fluxes": [False, False]}
    o = np.arange(2, 7)

    ### param_grids["2nd"] =  odict(adv_order=dict(h_sca_adv_order=[2], v_sca_adv_order=[2], h_mom_adv_order=[2], v_mom_adv_order=[2]))
    param_grids["dz_out msf=1"] = odict(hybrid_opt=[0])
    param_grids["trb no_debug msf=1"] = odict(timing=dict(
        end_time=["2018-06-20_12:30:00"],
        output_streams=[{24: ["meanout", 2. / 60.], 0: ["instout", 10.]}]))
    param_grids["trb no_debug hor_avg msf=1"] = odict(timing=dict(
        end_time=["2018-06-20_12:30:00"],
        output_streams=[{24: ["meanout", 2. / 60.], 0: ["instout", 10.]}]))
    param_grids["hor_avg msf=1"] = odict(km_opt=[2])  # for Y=0 test
    param_grids["hor_avg"] = odict(hybrid_opt=[0])
    param_grids["hessel"] = odict(hesselberg_avg=[True, False])
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
    param_grids["open BC y hor_avg"] = odict(open_ys=[True], open_ye=[True], periodic_y=[False],
                                             hm=hm, spec_hfx=[0.2], input_sounding="free")
    param_grids["symmetric BC x"] = odict(symmetric_xs=[True], symmetric_xe=[True], periodic_x=[False],
                                          hm=hm, spec_hfx=[0.2], input_sounding="free")
    param_grids["symmetric BC y"] = odict(symmetric_ys=[True], symmetric_ye=[True], periodic_y=[False],
                                          hm=hm, spec_hfx=[0.2], input_sounding="free")
    param_grids["symmetric BC y hor_avg"] = odict(symmetric_ys=[True], symmetric_ye=[True], periodic_y=[False],
                                                  hm=hm, spec_hfx=[0.2], input_sounding="free")

    failed, failed_short, err, err_short = run_and_check_budget(param_grids, avg_dims=["y"])

    return failed, failed_short, err, err_short

# %%


def run_and_check_budget(param_grids, config_file="wrflux.test.config_test_tendencies", avg_dims=None):
    # %%run_and_check_budget

    index = pd.MultiIndex.from_product([["INIT", "RUN"] + tests, variables])
    failed = pd.DataFrame(index=index)
    failed_short = pd.DataFrame(columns=variables)
    index = pd.MultiIndex.from_product([tests, variables])
    err = pd.DataFrame(index=index)

    for label, param_grid in param_grids.items():
        print("\n\n\n{0}\nRun test simulations: {1}\n{0}\n".format("#" * 50, label))
        # initialize and run simulations
        debugs = debug
        if "no_debug" in label:
            debugs = False
        for deb in tools.make_list(debugs):
            cfile = config_file
            if deb:
                cfile = cfile + "_debug"
            conf = importlib.import_module(cfile)
            rmsf = random_msf
            if "msf=1" in label:
                rmsf = False
            setup_test_init_module(conf, debug=deb, random_msf=rmsf)
            param_combs = grid_combinations(param_grid, conf.params, param_names=conf.param_names,
                                            runID=conf.runID)
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
                    failed[ind].loc["INIT", :] = "F"
                    continue
                with open("{}/run.log".format(run_dir)) as f:
                    log = f.read()
                if "wrf: SUCCESS COMPLETE WRF" not in log:
                    print("Error in running simulation {}!".format(cname))
                    failed[ind].loc["RUN", :] = "F"
                    continue

        for cname, param_comb in param_combs.iterrows():
            if cname in ["core_param", "composite_idx"]:
                continue
            IDi = param_comb["fname"]
            ind = label + ": " + IDi
            if (failed.loc[["RUN", "INIT"], ind] == "F").any():
                continue
            print("\n\n\n{0}\nPostprocess simulation: {1}, {2}\n{0}\n".format("#" * 50, label, cname))

            failed[ind] = ""
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
                skip_exist=skip_exist, save_output=True, return_model_output=True, chunks=chunks)

            print("\n\n\n{0}\nRun tests\n{0}\n".format("#" * 50))
            if rmsf and ("Y=0" in tests):
                tests.remove("Y=0")

            failed_i, err_i = testing.run_tests(datout, dat_inst=dat_inst, tests=tests_i,
                                                hor_avg=hor_avg, trb_exp=t_avg, chunks=chunks, **kw)

            for var in variables:
                for test, f in failed_i.loc[var].items():
                    if f:
                        failed[ind].loc[test, var] = "F"

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
            failed_tests = ",".join([test for test, f in failed[ind].loc[:, var].iteritems() if f == "F"])
            failed_short.loc[ind, var] = failed_tests

    failed_short = failed_short.where(failed_short != "").dropna(how="all").dropna(axis=1, how="all")
    failed_short = failed_short.where(~failed_short.isnull(), "")
    err_short = err.where(failed == "F").dropna(how="all").dropna(axis=1, how="all")
    err_short = err_short.where(~err_short.isnull(), "")
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

    # request.addfinalizer(setup_test)

    # Code that will run before each test
    delete_test_data()
    # A test function will be run at this point
    yield
    # Code that will run after each test
    delete_test_data()


def delete_test_data():
    for d in [os.environ["wrf_res"] + "/test/pytest", os.environ["wrf_runs"] + "/test/pytest"]:
        if os.path.isdir(d):
            shutil.rmtree(d)


def capture_submit(*args, **kwargs):
    try:
        with Capturing() as output:
            combs = launch_jobs(*args, **kwargs)
    except Exception as e:
        print("\n".join(output))
        raise(e)

    return combs, output


def setup_test_init_module(conf, debug=False, restore=False, random_msf=True):
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
        path = os.path.realpath(__file__)
        test_path = path[:path.index("test_tendencies.py")]
        testf = "TEST_"
        if random_msf:
            testf += "msf_"
        with open(test_path + testf + fname) as f:
            test_file = f.read()
        if test_file == org_file:
            return
        else:
            m = "Copy"
            shutil.copy(fpath, fpath + ".org")
            shutil.copy(test_path + testf + fname, fpath)

        print(m + " module_initialize_ideal.F and recompile")
        os.chdir(wrf_path)
        os.system("./compile em_les > log 2> err")
        os.chdir(test_path)


# %%main
if __name__ == "__main__":
    failed, failed_short, err, err_short = test_all()
