#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:54:43 2020

Calculate the WRF budget terms for theta or qv under different assumptions.
Scatter and profile plots of the individual terms and sums.

@author: Matthias Göbel
"""
from run_wrf.submit_jobs import submit_jobs
import shutil
from run_wrf import misc_tools
from collections import OrderedDict as odict
from run_wrf.misc_tools import Capturing
from collections import Counter
import os
from wrflux import tools
import xarray as xr
import pandas as pd
import numpy as np
import pytest
import testing
import importlib
xr.set_options(arithmetic_join="exact")
xr.set_options(keep_attrs=True)

os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"

XY = ["X", "Y"]
exist = "o"

# thresh_thdry = 0.02#TODOm

variables = ["q", "t", "u", "v", "w"]
# variables = ["q"]

raise_error = False
skip_exist = True
debug = [True,False]
random_msf = False #change mapscale factors from 1 to random values

#TODO: sgs boundary values for open bc should be zero!
#TODO: hor avg needs boundary points?
#TODO: more tests: mean_flux: mean~ tot
#TODO: go through with debugger to find hidden errors and add comments

# t_avg = False #average over time
# t_avg_interval = 4 #size of the time averaging window

hor_avg = False#TODOm
avg_dims = ["y"]
plot = True
plot_kws = dict(
    # plot_diff = True, #plot difference between forcing and tendency against tendency
    discrete=True,
    # iloc={"bottom_top" : [0,1,2,3,-4,-3,-2,-1]},
    hue="dir",
    s=40,
    ignore_missing_hue=True,
    # savefig = False,
    close = True
    )
#%%set calculation methods
budget_methods = [
            [[], "native"],
            ["cartesian", "correct"],
         ]
budget_methods_2nd = [*budget_methods,
            ["cartesian", "2nd"],
            ["cartesian", "correct", "2nd"],
            ["cartesian"],
         ]
budget_methods_dzout = [*budget_methods,
                        ["cartesian", "correct", "dz_out"],
                        ["cartesian", "correct", "dz_out", "corr_varz"],]

if exist != "s":
    skip_exist = False
#%%test functions

def test_budget_all():
    import config_test_tendencies as conf

    setup_test_init_module(conf, random_msf=random_msf) #TODOm: also for debug!

    #Define parameter grid for simulations
    param_grids = {}
    th={"use_theta_m" : [0,1,1],  "output_dry_theta_fluxes" : [False,False,True]}
    th2={"use_theta_m" : [0,1],  "output_dry_theta_fluxes" : [False,False]}
    param_grids["hor_avg"] = odict(km_opt=[2])
    param_grids["dz_out"] = odict(spec_hfx=[None])
    param_grids["hessel"] = odict(hesselberg_avg=[True,False])
    param_grids["serial"] = odict(lx=[5000], ly=[5000])
    param_grids["km_opt"] = odict(km_opt=[2,5], spec_hfx=[0.2, None], th=th)
    param_grids["PBL scheme with theta moist/dry"] = odict(bl_pbl_physics=[1], th=th)
    o = np.arange(2,7)
    param_grids["simple and positive-definite advection"] = odict(moist_adv_opt=[0,1], adv_order=dict(h_sca_adv_order=o, v_sca_adv_order=o, h_mom_adv_order=o, v_mom_adv_order=o))
    param_grids["WENO advection"] = odict(moist_adv_opt=[0,3,4], scalar_adv_opt=[3], momentum_adv_opt=[3], th=th2)
    param_grids["monotonic advection"] = odict(moist_adv_opt=[2], v_sca_adv_order=[3,5], th=th2)
    param_grids["MP rad"] = odict(mp_physics=[2], th=th)

    failed = run_and_check_budget(param_grids, hor_avg=hor_avg, avg_dims=avg_dims)
    # setup_test_init_module(conf, restore=True)

    return failed

def test_budget_bc():
    import config_test_tendencies as conf
    setup_test_init_module(conf, random_msf=random_msf) #TODOm: also for debug!

    #Define parameter grid for simulations
    param_grids = {}
    param_grids["open BC x"] = odict(open_xs=[True],open_xe=[True],periodic_x=[False], spec_hfx=[0.2])
    param_grids["open BC y"] = odict(open_ys=[True],open_ye=[True],periodic_y=[False], spec_hfx=[0.2])
    param_grids["symmetric BC"] = odict(symmetric_ys=[True],symmetric_ye=[True],periodic_y=[False], spec_hfx=[0.2])

    failed = run_and_check_budget(param_grids, conf, hor_avg=hor_avg, avg_dims=avg_dims)
    # setup_test_init_module(conf, restore=True)

    return failed

def run_and_check_budget(param_grids, config_file="wrflux.test.config_test_tendencies", hor_avg=False, avg_dims=None):
    #%%run_and_check_budget
    failed = pd.DataFrame(columns=variables)

    for label, param_grid in param_grids.items():
        print("\n\n\nTest " + label)
        #initialize and run simulations
        for deb in tools.make_list(debug):
            cfile = config_file
            if deb:
                cfile = cfile + "_debug"
            conf = importlib.import_module(cfile)
            param_combs = misc_tools.grid_combinations(param_grid, conf.params, param_names=conf.param_names, runID=conf.runID)
            combs, output = capture_submit(init=True, exist=exist, debug=deb, config_file=cfile, param_combs=param_combs)
            combs, output = capture_submit(init=False, wait=True, debug=deb, pool_jobs=True, exist=exist,
                                           config_file=cfile, param_combs=param_combs)
            print("\n\n")

            for cname, param_comb in param_combs.iterrows():
                if cname in ["core_param", "composite_idx"]:
                    continue
                IDi = param_comb["fname"]
                #check logs
                run_dir = os.path.join(conf.run_path, "WRF_" + IDi + "_0")
                with open("{}/init.log".format(run_dir)) as f:
                    log = f.read()
                if "wrf: SUCCESS COMPLETE IDEAL INIT" not in log:
                    print("Error in initializing simulation {}!".format(cname))
                    failed.loc[IDi, :] = "INIT "
                    continue
                with open("{}/run.log".format(run_dir)) as f:
                    log = f.read()
                if "wrf: SUCCESS COMPLETE WRF" not in log:
                    print("Error in running simulation {}!".format(cname))
                    failed.loc[IDi, :] = "RUN "
                    continue

        for cname, param_comb in param_combs.iterrows():
            if cname in ["core_param", "composite_idx"]:
                continue
            IDi = param_comb["fname"]
            print("\n\n\n{0}\nPostprocess simulation: {1}\n{0}\n".format("#"*50, cname))

            if IDi not in failed.index:
                failed.loc[IDi] = ""
            #postprocessing
            print("Postprocess data")
            adv_2nd = False
            dzout = False
            bm = budget_methods
            hor_avg_i = hor_avg
            if label == "hor_avg":
                #test hor_avg
                hor_avg_i = True
            elif label == "dz_out":
                #test dzout
                dzout = True
                bm = budget_methods_dzout
            elif param_comb["h_sca_adv_order"] == param_comb["v_sca_adv_order"] == param_comb["h_mom_adv_order"] == param_comb["v_mom_adv_order"] == 2:
                adv_2nd = True
                bm = budget_methods_2nd

            outpath_c = os.path.join(conf.outpath, IDi) + "_0"
            datout, dat_inst, dat_mean = tools.calc_tendencies(variables, outpath_c, start_time=param_comb["start_time"], budget_methods=bm,
                                  skip_exist=skip_exist, hor_avg=hor_avg_i, avg_dims=avg_dims, save_output=True)

            print("\n\n\n{0}\nRun tests\n{0}\n".format("#"*50))

            for var in variables:
                print("Variable: " + var)
                datout_v = datout[var]
                # cut_boundaries_c = cut_boundaries TODOm
                # if not attrs["PERIODIC_X"]:
                #     bounds["x"] = slice(b,-b)
                #     cut_boundaries_c = True
                # if not attrs["PERIODIC_Y"]:
                #     bounds["y"] = slice(b,-b)
                #     cut_boundaries_c = True
                # if cut_boundaries_c:
                #     bounds_v = create_bounds(forcing)
                #     forcing = forcing[bounds_v]
                #     total_tend = total_tend[bounds_v]
                #tests
                failed_i = {}
                failed_i["budget"] = testing.test_budget(datout_v["tend"], thresh=0.01, plot=plot, **plot_kws)
                adv = datout_v["adv"]
                corr = datout_v["corr"]
                if var == "w":#TODOm
                    adv = adv.isel(bottom_top_stag=slice(0,-1))
                    corr = corr.isel(bottom_top_stag=slice(0,-1))
                else:
                    adv = adv.isel(bottom_top=slice(0,-1))
                    corr = corr.isel(bottom_top=slice(0,-1))

                failed_i["decomp_sumdir"] = testing.test_decomp_sumdir(adv, corr, thresh=0.04, plot=True, **plot_kws)
                failed_i["decomp_sumcomp"] = testing.test_decomp_sumcomp(datout_v["adv"], thresh=0.01, plot=True, **plot_kws)
                if dzout:#TODOm: why limit?
                    failed_i["dz_out"] = testing.test_dz_out(adv, plot=plot, **plot_kws)
                if adv_2nd:
                    failed_i["2nd"] = testing.test_2nd(datout_v["adv"], thresh=0.01, plot=plot, **plot_kws)
                failed_i["NaN"] = testing.test_nan(datout_v)
                if var == variables[-1]:
                    failed_i["w"] = testing.test_w(dat_inst.isel(Time=slice(1,None)), thresh=0.015, plot=plot, **plot_kws)

                failed.loc[IDi, var] += ",".join([key for key, val in failed_i.items() if val])

    if (failed != "").values.any():
        message = "\n\n{}\nFailed tests:\n{}".format("#"*100,failed.to_string())
        if raise_error:
            raise RuntimeError(message)
        else:
            print(message)

    return failed


#%% misc
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
    for d in [os.environ["wrf_res"] + "/test/pytest", os.environ["wrf_runs"] + "/pytest"]:
        if os.path.isdir(d):
            shutil.rmtree(d)

def capture_submit(*args, **kwargs):
    try:
        with Capturing() as output:
            combs = submit_jobs(*args, **kwargs)
    except Exception as e:
        print("\n".join(output))
        raise(e)

    return combs, output

def setup_test_init_module(conf, restore=False, random_msf=True):
    fname = "module_initialize_ideal.F"
    wrf_path = "{}/{}".format(conf.build_path, conf.serial_build)
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

        print(m +  " module_initialize_ideal.F and recompile")
        os.chdir(wrf_path)
        os.system("./compile em_les > log 2> err")

#%%main
if __name__ == "__main__":
    failed = test_budget_all()
    #failed_bc = test_budget_bc()
