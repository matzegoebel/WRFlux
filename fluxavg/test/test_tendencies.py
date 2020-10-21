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
exist = "o"
debug = True
thresh_thdry = 0.02
thresh = 0.003
cut_boundaries = False
cartesian = True
plot = True
#%%settings

def test_budget(exist="s", debug=False, thresh=0.02, thresh_thdry=0.002,
                cartesian=True, plot=False):
#%%

    setup_test_init_module()

    #Define parameter grid for simulations
    param_grids = {}
    thm={"use_theta_m" : [0,1,1],  "output_dry_theta_fluxes" : [False,False,True]}
    param_grids["km_opt"] = odict(km_opt=[2,5], spec_hfx=[0.2, None], thm=thm)
    # param_grids["PBL schemes"] = odict(bl_pbl_physics=[*np.arange(1,13), 99])
    param_grids["PBL scheme with theta moist/dry"] = odict(bl_pbl_physics=[1], thm=thm)
    o = np.arange(2,7)
    param_grids["simple and positive-definite advection"] = odict(moist_adv_opt=[0,1], adv_order=dict(h_sca_adv_order=o, v_sca_adv_order=o, h_mom_adv_order=o, v_mom_adv_order=o))
    param_grids["WENO advection"] = odict(moist_adv_opt=[0,3,4], scalar_adv_opt=[3], momentum_adv_opt=[3])
    param_grids["monotonic advection"] = odict(moist_adv_opt=[2], v_sca_adv_order=[3,5])
    param_grids["MP rad"] = odict(mp_physics=[2])

    failed = {}
    for label, param_grid in param_grids.items():
        print("\n\n\nTest " + label)
        param_combs = misc_tools.grid_combinations(param_grid, conf.params, param_names=conf.param_names, runID=conf.runID)
        #initialize and run simulations
        combs, output = capture_submit(init=True, exist=exist, debug=debug,
                                       config_file="config_test", param_combs=param_combs)
        c = Counter(output)
        if c['wrf: SUCCESS COMPLETE IDEAL INIT'] + c['Skipping...'] != len(combs):
            raise RuntimeError("Error in initializing simulations!")
        combs, output = capture_submit(init=False, wait=True, pool_jobs=True, exist=exist,
                                       config_file="config_test", param_combs=param_combs)
        c = Counter(output)
        if c['d01 {} wrf: SUCCESS COMPLETE WRF'.format(conf.params["end_time"])] + c['Skipping...'] != len(combs):
            raise RuntimeError("Error in running simulations!")
        print("\n\n")
        #% postprocessing
        for cname, comb in combs.iterrows():
            IDi = comb["fname"]
            print("Run: {} \n".format(cname))
            dat_mean, dat_inst = load_data(IDi)
            dat_mean, dat_inst, grid, cyclic, stagger_const, attrs = tools.prepare(dat_mean, dat_inst)
            for var in ["q", "th", "u", "v", "w"]:
                forcing, total_tend = get_tendencies(var, dat_inst, dat_mean, grid, cyclic, stagger_const, attrs,
                                                     cartesian=cartesian, correct=True, recalc_w=True)
                if cut_boundaries:
                    forcing = forcing[:,1:-1,1:-1,1:-1]
                    total_tend = total_tend[:,1:-1,1:-1,1:-1]
                value_range = total_tend.max() - total_tend.min()
                max_error = abs(forcing - total_tend).max()
                e = max_error/value_range
                if plot:
                    tools.scatter_tend_forcing(total_tend, forcing, var, cut_boundaries=cut_boundaries, savefig=False)
                print("{0}: {1:.3f}%".format(var, e.values*100))
                if (attrs["USE_THETA_M"] == 1) and (attrs["OUTPUT_DRY_THETA_FLUXES"] == 1) and (var == "th"):
                    thresh_v = thresh_thdry
                else:
                    thresh_v = thresh
                if e > thresh_v:
                    print("Variable {} FAILED!".format(var))
                    if cname in failed:
                        failed[cname].append(var)
                    else:
                        failed[cname] = [var]
#%%
    if failed != {}:
        raise RuntimeError("Forcing unequal total tendency for following runs/variables:\n{}"\
                           .format("\n".join(["{} : {}".format(k,v) for k,v in failed.items()])))
  #  setup_test_init_module(restore=True)

                #TODO: more tests: mean_flux,..., which input_sounding?
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

def capture_submit(*args, **kwargs):
    try:
        with Capturing() as output:
            combs = submit_jobs(*args, **kwargs)
    except Exception as e:
        print(output)
        raise(e)

    return combs, output

def setup_test_init_module(restore=False):
    fpath = os.path.realpath(__file__)
    test_path = fpath[:fpath.index("test_tendencies.py")]
    fname = "module_initialize_ideal.F"
    wrf_path = fpath[:fpath.index("/fluxavg/")]
    fpath = wrf_path + "/dyn_em/" + fname
    with open(fpath) as f:
        org_file = f.read()
    with open(test_path + fname + ".test") as f:
        test_file = f.read()
    if (test_file != org_file) or restore:
        if restore:
            shutil.copy(fpath + ".org", fpath)
        else:
            shutil.copy(fpath, fpath + ".org")
            shutil.copy(test_path + fname + ".test", fpath)
        os.chdir(code_path)
        os.system("./compile em_les > log 2> err")


#%%
if __name__ == "__main__":
    test_budget(exist=exist, debug=debug, thresh=thresh, thresh_thdry=thresh_thdry,
                cartesian=cartesian, plot=plot)