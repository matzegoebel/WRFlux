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
import config_test as conf
import pandas as pd
import numpy as np
import pytest
xr.set_options(arithmetic_join="exact")
xr.set_options(keep_attrs=True)

os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"

XY = ["X", "Y"]
outpath = os.path.join(conf.outpath, conf.outdir)
exist = "s"

thresh_thdry = 0.02
thresh = 0.003

variables = ["q", "t", "u", "v", "w"]
# variables = ["q"]
cut_boundaries = False
bounds = {"x" : slice(1,-1), "y" : slice(1,-1), "bottom_top" : slice(1,-1)}
bounds = {"bottom_top" : slice(None,-1)}
cartesian = True
plot = False
plot_diff = False
raise_error = False
nan_check = True
#TODO: automatically cut_bounds if bc are not symmetric/periodic
#TODO: why so bad correlation for openbc
#TODO: sgs boundary values for open bc should be zero!
#%%settings

def test_budget():
#%%

    setup_test_init_module()

    #Define parameter grid for simulations
    param_grids = {}
    thm={"use_theta_m" : [0,1,1],  "output_dry_theta_fluxes" : [False,False,True]}
    # param_grids["km_opt"] = odict(km_opt=[2], spec_hfx=[0.2])
    # param_grids["open BC"] = odict(open_xs=[True],open_xe=[True],periodic_x=[False], km_opt=[2], use_theta_m=[0])
    param_grids["km_opt"] = odict(km_opt=[2,5], spec_hfx=[0.2, None], thm=thm)
    param_grids["PBL scheme with theta moist/dry"] = odict(bl_pbl_physics=[1], thm=thm)
    # param_grids["PBL schemes"] = odict(bl_pbl_physics=[*np.arange(1,13), 99])
    o = np.arange(2,7)
    param_grids["simple and positive-definite advection"] = odict(moist_adv_opt=[0,1], adv_order=dict(h_sca_adv_order=o, v_sca_adv_order=o, h_mom_adv_order=o, v_mom_adv_order=o))
    param_grids["WENO advection"] = odict(moist_adv_opt=[0,3,4], scalar_adv_opt=[3], momentum_adv_opt=[3], thm=thm)
    param_grids["monotonic advection"] = odict(moist_adv_opt=[2], v_sca_adv_order=[3,5], thm=thm)
    param_grids["MP rad"] = odict(mp_physics=[2])

    failed = {}
    error = pd.DataFrame(columns=variables)
    for label, param_grid in param_grids.items():
        print("\n\n\nTest " + label)
        param_combs = misc_tools.grid_combinations(param_grid, conf.params, param_names=conf.param_names, runID=conf.runID)
        #initialize and run simulations
        combs, output = capture_submit(init=True, exist=exist, config_file="config_test", param_combs=param_combs)
        combs, output = capture_submit(init=False, wait=True, pool_jobs=True, exist=exist,
                                       config_file="config_test", param_combs=param_combs)
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
            dat_mean, dat_inst = load_data(IDi)
            dat_mean, dat_inst, grid, cyclic, stagger_const, attrs = tools.prepare(dat_mean, dat_inst)
            for var in variables:
                # check_bounds(dat_mean, attrs, var) #TODO: fix for open
                forcing, total_tend, adv, sgs, sources, fluxes, corr = get_tendencies(var, dat_inst, dat_mean,
                        grid, cyclic, stagger_const, attrs, cartesian=cartesian, correct=True, recalc_w=True)
                if cut_boundaries:
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
                    check_nans(total_tend, adv, sgs, sources, fluxes, var)


    print("\n\nMaximum absolute tendency reconstruction error normalized by tendency range in %:\n{}".format(error))
#%%
    if failed != {}:
        message = "Reconstructed forcing deviates strongly from actual tendency for following runs/variables:\n{}"\
                           .format("\n".join(["{} : {}".format(k,v) for k,v in failed.items()]))
        if raise_error:
            raise RuntimeError(message)
        else:
            print(message)
    #setup_test_init_module(restore=True)

                #TODO: more tests: mean_flux: mean~ tot, test hesselberg, test other BC
    return error, failed
#%% postprocess data
def load_data(IDi):
    dat_inst = tools.open_dataset(outpath + "/instout_{}_0".format(IDi), del_attrs=False)
    dat_mean = tools.open_dataset(outpath + "/meanout_{}_0".format(IDi))

    const_vars = ["ZNW", "ZNU", "DN", "DNW", "FNM", "FNP", "CFN1", "CFN", "CF1", "CF2", "CF3", "C1H", "C2H", "C1F", "C2F"]
    for v in dat_inst.data_vars:
        if (v in const_vars) or ("MAPFAC" in v):
            dat_inst[v] = dat_inst[v].sel(drop=True)

    return dat_mean, dat_inst

def get_tendencies(var, dat_inst, dat_mean, grid, cyclic, stagger_const, attrs, cartesian=False, correct=True, recalc_w=True):
    VAR = var.upper()

    dat_mean, dat_inst, total_tend, sgs, sources, grid, dim_stag, mapfac, dzdd, dzdd_s \
     = tools.calc_tend_sources(dat_mean, dat_inst, var, grid, cyclic, stagger_const, attrs)

    var_stag = xr.Dataset()
    #get staggered variables
    for d in [*XY, "Z"]:
        var_stag[d] = dat_mean["{}{}_MEAN".format(VAR, d)]

    flux, adv, vmean = tools.adv_tend(dat_mean, VAR, var_stag, grid, mapfac, cyclic, stagger_const, cartesian=cartesian,
                                          recalc_w=recalc_w)
    corr = None
    if correct and cartesian:
        corr = dat_mean[["F{}X_CORR".format(VAR), "F{}Y_CORR".format(VAR), "CORR_D{}DT".format(VAR)]].to_array("dim")
        adv, total_tend, corr = tools.cartesian_corrections(VAR, dim_stag, corr, var_stag, vmean, dat_mean["RHOD_MEAN"],
                                                      dzdd, grid, adv, total_tend, cyclic, stagger_const)

    #add all forcings
    forcing = adv.sel(comp="tot", drop=True).sum("dir") + sources

    return forcing, total_tend, adv, sgs, sources, flux, corr

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

def check_nans(total_tend, adv, sgs, sources, fluxes, var):
    for n, dat in zip(["total tendency", "advection", "SGS diffusion", "sources", "X flux", "Y flux", "Z flux"], [total_tend, adv, sgs, sources, fluxes["X"], fluxes["X"], fluxes["Z"]]):
        if (n != "total tendency") and cut_boundaries:
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

def setup_test_init_module(restore=False):
    fpath = os.path.realpath(__file__)
    test_path = fpath[:fpath.index("test_tendencies.py")]
    fname = "module_initialize_ideal.F"
    wrf_path = "{}/{}".format(conf.build_path, conf.wrf_dir_pre)
    fpath = wrf_path + "/dyn_em/" + fname
    with open(fpath) as f:
        org_file = f.read()
    with open(test_path + "TEST_" + fname) as f:
        test_file = f.read()
    if (test_file != org_file) or restore:
        if restore:
            shutil.copy(fpath + ".org", fpath)
        else:
            shutil.copy(fpath, fpath + ".org")
            shutil.copy(test_path + "TEST_" + fname, fpath)
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
#%%
if __name__ == "__main__":
    error, failed = test_budget()