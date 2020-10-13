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
import os
import tools
import xarray as xr
import config_test as conf
import numpy as np
xr.set_options(arithmetic_join="exact")
xr.set_options(keep_attrs=True)

os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"

XY = ["X", "Y"]
outpath = os.path.join(conf.outpath, conf.outdir)
exist = "s"
debug = False
#%%settings

def test_budget(exist="s", debug=False):

    #Define parameter grid for simulations
    param_grids = {}
    thm={"use_theta_m" : [0,1,1],  "output_dry_theta_fluxes" : [False,False,True]}
    param_grids["km_opt"] = odict(km_opt=[2,5], spec_hfx=[0.2, None], thm=thm)
    param_grids["PBL schemes"] = odict(bl_pbl_physics=[*np.arange(1,13), 99])
    param_grids["PBL scheme with theta moist/dry"] = odict(bl_pbl_physics=[1], thm=thm)
    o = np.arange(2,7)
    param_grids["simple and positive-definite advection"] = odict(moist_adv_opt=[0,1], adv_order=dict(h_sca_adv_order=o, v_sca_adv_order=o, h_mom_adv_order=o, v_mom_adv_order=o))
    param_grids["WENO advection"] = odict(moist_adv_opt=[3,4], momentum_adv_opt=[3])
    param_grids["monotonic advection"] = odict(moist_adv_opt=[2], v_sca_adv_order=[3,5])

    for label, param_grid in param_grids.items():
        print("Test " + label)
        param_combs, combs, param_grid_flat, composite_params = misc_tools.grid_combinations(param_grid, conf.params)
        #initialize and run simulations
        kw = dict(param_grid=param_grid, param_combs=param_combs, combs=combs, composite_params=composite_params)
        submit_jobs(init=True, exist=exist, debug=debug, config_file="config_test", **kw)
        combs = submit_jobs(init=False, wait=True, pool_jobs=True, exist=exist, config_file="config_test", **kw)

        print("\n\n\n")
        #postprocessing
        for i in range(len(combs)):
            IDi, IDi_d = misc_tools.output_id_from_config(param_combs[i], param_grid, conf.param_names, conf.runID)
            print("Run: {} \n".format(IDi_d))
            dat_mean, dat_inst = load_data(IDi)
            dat_mean, dat_inst, grid, cyclic, stagger_const, attrs = tools.prepare(dat_mean, dat_inst)
            for var in ["q", "th", "u", "v", "w"]:
                print("Variable: " + var)
                forcing, total_tend = get_tendencies(var, dat_inst, dat_mean, grid, cyclic, stagger_const, attrs, cartesian=False, correct=True, recalc_w=True)
                value_range = total_tend.max() - total_tend.min()
                max_error = abs(forcing - total_tend).max()
                assert max_error/value_range < 0.02
                #TODO: more tests: mean_flux,...

#%%
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

    if correct and cartesian:
        corr = dat_mean[["CORR_U{}".format(VAR), "CORR_V{}".format(VAR), "CORR_D{}DT".format(VAR)]].to_array("dim")
        adv, total_tend = tools.cartesian_corrections(VAR, dim_stag, corr, var_stag, vmean, dat_mean["RHOD_MEAN"],
                                                      dzdd, grid, adv, total_tend, cyclic, stagger_const)

    #add all forcings
    forcing = adv["Z"].sel(comp="tot", drop=True) + sources
    for d in XY:
        forcing = forcing + adv[d].sel(comp="tot", drop=True)

    return forcing, total_tend

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
#%%
if __name__ == "__main__":
    test_budget(exist=exist, debug=debug)