#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:54:43 2020

Calculate the WRF budget terms for theta or qv under different assumptions.
Scatter and profile plots of the individual terms and sums.

@author: c7071088
"""
from run_wrf.submit_jobs import submit_jobs
import shutil
from run_wrf import misc_tools
from collections import OrderedDict as odict
from run_wrf.misc_tools import Capturing
from collections import Counter
import matplotlib.pyplot as plt
import os
import tools
import xarray as xr
import pandas as pd
import numpy as np
import pytest
xr.set_options(arithmetic_join="exact")
xr.set_options(keep_attrs=True)

os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"

XY = ["X", "Y"]
exist = "s"

thresh_thdry = 0.02
thresh = 0.01

variables = ["q", "t", "u", "v", "w"]
# variables = ["w"]
cut_boundaries = False
b = 1
bounds = {"x" : slice(b,-b), "y" : slice(b,-b), "bottom_top" : slice(b,-b)}
# bounds = {"x" : slice(b,-b)}
# bounds = {"bottom_top" : slice(None,-b)}
cartesian = False
plot = True
plot_diff = False
raise_error = False
nan_check = True
#TODO: sgs boundary values for open bc should be zero!
#TODO: hor avg needs boundary points?
#TODO: more tests: mean_flux: mean~ tot, test hesselberg
#TODO: go through with debugger to find hidden errors and add comments
#%%test functions

def test_decomp():
    import config_test_decomp as conf
    config_file="config_test_decomp"
     #Define parameter grid for simulations
    param_grids = {}
    th={"use_theta_m" : [0,1,1],  "output_dry_theta_fluxes" : [False,False,True]}
    th2={"use_theta_m" : [0,1],  "output_dry_theta_fluxes" : [False,False]}
    param_grids["km_opt"] = odict(hesselberg_avg=[True])
    # param_grids["km_opt"] = odict(spec_hfx=[0.2])
    # param_grids["km_opt"] = odict(km_opt=[2,5], spec_hfx=[0.2, None], th=th)
    # param_grids["PBL scheme with theta moist/dry"] = odict(bl_pbl_physics=[1], th=th)
    # o = np.arange(2,7)
    # param_grids["simple and positive-definite advection"] = odict(moist_adv_opt=[0,1], adv_order=dict(h_sca_adv_order=o, v_sca_adv_order=o, h_mom_adv_order=o, v_mom_adv_order=o))
    # param_grids["WENO advection"] = odict(moist_adv_opt=[0,3,4], scalar_adv_opt=[3], momentum_adv_opt=[3], th2=th2)
    # param_grids["monotonic advection"] = odict(moist_adv_opt=[2], v_sca_adv_order=[3,5], th2=th2)
    hor_avg = True
    avg_dims = ["x"]

    error, failed = run_and_check_budget(param_grids, conf, config_file=config_file,
                                         hor_avg=hor_avg, avg_dims=avg_dims)

    return error, failed

def test_budget_all():
    import config_test_tendencies as conf

    setup_test_init_module(conf)

    #Define parameter grid for simulations
    param_grids = {}
    th={"use_theta_m" : [0,1,1],  "output_dry_theta_fluxes" : [False,False,True]}
    th2={"use_theta_m" : [0,1],  "output_dry_theta_fluxes" : [False,False]}
    param_grids["km_opt"] = odict(hesselberg_avg=[True,False])
    param_grids["km_opt"] = odict(km_opt=[2,5], spec_hfx=[0.2, None], th=th)
    param_grids["PBL scheme with theta moist/dry"] = odict(bl_pbl_physics=[1], th=th)
    o = np.arange(2,7)
    param_grids["simple and positive-definite advection"] = odict(moist_adv_opt=[0,1], adv_order=dict(h_sca_adv_order=o, v_sca_adv_order=o, h_mom_adv_order=o, v_mom_adv_order=o))
    param_grids["WENO advection"] = odict(moist_adv_opt=[0,3,4], scalar_adv_opt=[3], momentum_adv_opt=[3], th2=th2)
    param_grids["monotonic advection"] = odict(moist_adv_opt=[2], v_sca_adv_order=[3,5], th2=th2)
    param_grids["MP rad"] = odict(mp_physics=[2], th=th)

    error, failed = run_and_check_budget(param_grids, conf)
    setup_test_init_module(conf, restore=True)

    return error, failed

def test_budget_bc():
    import config_test_tendencies as conf

    #Define parameter grid for simulations
    param_grids = {}
    param_grids["open BC x"] = odict(open_xs=[True],open_xe=[True],periodic_x=[False], spec_hfx=[0.2])
    param_grids["open BC y"] = odict(open_ys=[True],open_ye=[True],periodic_y=[False], spec_hfx=[0.2])
    param_grids["symmetric BC"] = odict(symmetric_ys=[True],symmetric_ye=[True],periodic_y=[False], spec_hfx=[0.2])

    error, failed = run_and_check_budget(param_grids, conf)

    return error, failed

def run_and_check_budget(param_grids, conf, config_file="config_test_tendencies", hor_avg=False, avg_dims=None):
    #%%run_and_check_budget
    failed = {}
    error = pd.DataFrame(columns=variables)
    for label, param_grid in param_grids.items():
        print("\n\n\nTest " + label)
        param_combs = misc_tools.grid_combinations(param_grid, conf.params, param_names=conf.param_names, runID=conf.runID)
        #initialize and run simulations
        combs, output = capture_submit(init=True, exist=exist, config_file=config_file, param_combs=param_combs)
        combs, output = capture_submit(init=False, wait=True, pool_jobs=True, exist=exist,
                                       config_file=config_file, param_combs=param_combs)
        print("\n\n")
        #% postprocessing
        for cname, comb in combs.iterrows():
            IDi = comb["fname"]
            print("\n Run: {}".format(cname))
            with open("{}_0/init.log".format(comb["run_dir"])) as f:
                log = f.read()
            if "wrf: SUCCESS COMPLETE IDEAL INIT" not in log:
                print("Error in initializing simulations!")
                continue
            with open("{}_0/run.log".format(comb["run_dir"])) as f:
                log = f.read()
            if "wrf: SUCCESS COMPLETE WRF" not in log:
                print("Error in running simulations!")
                continue
            dat_mean, dat_inst = load_data(os.path.join(conf.outpath, IDi), comb["start_time"])
            dat_mean, dat_inst, grid, cyclic, attrs = tools.prepare(dat_mean, dat_inst)
            for var in variables:
                # check_bounds(dat_mean, attrs, var) #TODO: fix for open
                forcing, total_tend, adv, sgs, sgsflux, sources, fluxes, corr = get_tendencies(var, dat_inst, dat_mean,
                        grid, cyclic, attrs, cartesian=cartesian, correct=True,
                        hor_avg=hor_avg, avg_dims=avg_dims)

                cut_boundaries_c = cut_boundaries
                if not attrs["PERIODIC_X"]:
                    bounds["x"] = slice(b,-b)
                    cut_boundaries_c = True
                if not attrs["PERIODIC_Y"]:
                    bounds["y"] = slice(b,-b)
                    cut_boundaries_c = True
                if cut_boundaries_c:
                    bounds_v = create_bounds(forcing)
                    forcing = forcing[bounds_v]
                    total_tend = total_tend[bounds_v]

                check_error(total_tend, forcing, attrs, var, cname, error, failed)
                if var in ["t", "q"]:
                    r = np.corrcoef(dat_mean["WD_MEAN"].values.flatten(), dat_mean["W_MEAN"].values.flatten())[0,1]
                    if r < 0.9:
                        print("WARNING: diagnostic and prognostic w are poorly correlated!")
                #check for nans
                if nan_check:
                    check_nans(total_tend, adv, sgs, sgsflux, sources, fluxes, var, cut_boundaries=cut_boundaries_c)


    print("\n\nMaximum absolute tendency reconstruction error normalized by tendency range in %:\n{}\n\n".format(error.to_string()))
    if failed != {}:
        message = "Reconstructed forcing deviates strongly from actual tendency for following runs/variables:\n{}"\
                           .format("\n".join(["{} : {}".format(k,v) for k,v in failed.items()]))
        if raise_error:
            raise RuntimeError(message)
        else:
            print(message)
#%%
    return error, failed
#%% postprocess data
def load_data(outpath, start_time):
    dat_inst = tools.open_dataset("{}_0/instout_d01_{}".format(outpath, start_time), del_attrs=False)
    dat_mean = tools.open_dataset("{}_0/meanout_d01_{}".format(outpath, start_time))

    const_vars = ["ZNW", "ZNU", "DN", "DNW", "FNM", "FNP", "CFN1", "CFN", "CF1", "CF2", "CF3", "C1H", "C2H", "C1F", "C2F"]
    for v in dat_inst.data_vars:
        if (v in const_vars) or ("MAPFAC" in v):
            dat_inst[v] = dat_inst[v].sel(drop=True)

    return dat_mean, dat_inst

def get_tendencies(var, dat_inst, dat_mean, grid, cyclic, attrs, cartesian=False,
                   correct=True, hor_avg=False, avg_dims=None):
    VAR = var.upper()

    dat_mean, dat_inst, total_tend, sgs, sgsflux, sources, sources_sum, var_stag, grid, dim_stag, mapfac, dzdd \
     = tools.calc_tend_sources(dat_mean, dat_inst, var, grid, cyclic, attrs, hor_avg=hor_avg, avg_dims=avg_dims)

    flux, adv, vmean = tools.adv_tend(dat_mean, VAR, var_stag, grid, mapfac, cyclic, cartesian=cartesian,
                                      hor_avg=hor_avg, avg_dims=avg_dims)
    corr = None
    if correct and cartesian:
        corr = dat_mean[["F{}X_CORR".format(VAR), "F{}Y_CORR".format(VAR), "CORR_D{}DT".format(VAR)]]
        adv, total_tend, corr = tools.cartesian_corrections(VAR, dim_stag, corr, var_stag, vmean, dat_mean["RHOD_MEAN"],
                                                      dzdd, grid, mapfac, adv, total_tend, cyclic,
                                                      hor_avg=hor_avg, avg_dims=avg_dims)

    #add all forcings
    forcing = adv.sel(comp="adv_r", drop=True).sum("dir") + sources_sum

    return forcing, total_tend, adv, sgs, sgsflux, sources, flux, corr

#%% tests
def check_error(total_tend, forcing, attrs, var, cname, error, failed):
    value_range = total_tend.max() - total_tend.min()
    max_error = abs(forcing - total_tend).max()
    e = max_error/value_range
    error.loc[cname, var] = np.round(e.values*100,4)
    if (attrs["USE_THETA_M"] == 1) and (attrs["OUTPUT_DRY_THETA_FLUXES"] == 1) and (var == "t"):
        thresh_v = thresh_thdry
    else:
        thresh_v = thresh
    if e > thresh_v:
        if plot:
            fig = tools.scatter_tend_forcing(total_tend, forcing, var, plot_diff=plot_diff, savefig=False)
            fig.suptitle(cname)
        if cname in failed:
            failed[cname].append(var)
        else:
            failed[cname] = [var]

def check_nans(total_tend, adv, sgs, sgsflux, sources, fluxes, var, cut_boundaries=False):
    for n in ["total_tend", "adv", "sgs", "sources",
              "sgsflux['X']", "sgsflux['Y']", "sgsflux['Z']", "fluxes['X']", "fluxes['Y']", "fluxes['Z']"]:
        dat = eval(n)
        if (n != "total tend") and cut_boundaries:
            bounds_v = create_bounds(dat)
            dat = dat[bounds_v]
        dat = tools.find_nans(dat)
        if len(dat) != 0:
            print("\nWARNING: found NaNs in {} for variable {}:\n{}".format(n, var, dat.coords))

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
#%% misc
@pytest.fixture(autouse=True)
def run_around_tests():
    """Delete test data before and after every test"""

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

def setup_test_init_module(conf, restore=False):
    fpath = os.path.realpath(__file__)
    test_path = fpath[:fpath.index("test_tendencies.py")]
    fname = "module_initialize_ideal.F"
    wrf_path = "{}/{}".format(conf.build_path, conf.serial_build)
    fpath = wrf_path + "/dyn_em/" + fname
    with open(fpath) as f:
        org_file = f.read()
    with open(test_path + "TEST_" + fname) as f:
        test_file = f.read()
    if (test_file != org_file) or restore:
        if restore:
            m = "Restore"
            shutil.copy(fpath + ".org", fpath)
        else:
            m = "Copy"
            shutil.copy(fpath, fpath + ".org")
            shutil.copy(test_path + "TEST_" + fname, fpath)
        print(m +  " module_initialize_ideal.F and recompile")
        os.chdir(wrf_path)
        os.system("./compile em_les > log 2> err")

def create_bounds(data):
    bounds_v = bounds.copy()
    for d in ["x", "y", "bottom_top"]:
        if d not in bounds:
            bounds_v[d] = slice(None)
        if d not in data.dims:
            bounds_v[d + "_stag"] = bounds_v[d]
            del bounds_v[d]
    return bounds_v
#%%main
if __name__ == "__main__":
    error, failed = test_budget_all()
    error_bc, failed_bc = test_budget_bc()
    # error_d, failed_d = test_decomp()
