#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for launch_jobs.py
Test settings for automated tests.

@author: Matthias Göbel

"""
import os
from pathlib import Path
import numpy as np
from run_wrf.configs.base_config import *
from copy import deepcopy
params = deepcopy(params)


# %%
'''Simulations settings'''

runID = "pytest"
test_path = Path(__file__).parent.parent
test_sims = test_path / "test_sims"
params["outpath"] = str(test_sims / "results")  # WRF output path root
params["run_path"] = str(test_sims / "runs")  # path where run directories of simulations will be created

# path where different versions of the compiled WRF model code reside
build_path = str(test_path.parents[3])
params["build_path"] = build_path
params["serial_build"] = "WRFlux"  # used if nslots=1
params["parallel_build"] = "WRFlux"  # used if nslots > 1
params["debug_build"] = "WRFlux_debug"  # used for -d option
org_build = "WRF_org"  # original WRF version for comparison tests

# Fill dictionary params with default values to be used for parameters not present in param_grid

params["start_time"] = "2018-06-20_12:00:00"  # format %Y-%m-%d_%H:%M:%S
params["end_time"] = "2018-06-20_12:02:00"  # format %Y-%m-%d_%H:%M:%S

params["n_rep"] = 1  # number of repetitions for each configuration

# horizontal grid
params["dx"] = 200  # horizontal grid spacing x-direction(m)
params["dy"] = None  # horizontal grid spacing y-direction (m), if None: dy = dx
params["lx"] = 4000  # minimum horizontal extent in east west (m)
params["ly"] = 4000  # minimum horizontal extent in north south (m)

# control vertical grid creation (see vertical_grid.py for details on the different methods)
params["ztop"] = 3000  # top of domain (m)
params["zdamp"] = int(params["ztop"] / 5)  # depth of damping layer (m)
params["damp_opt"] = 0
params["nz"] = None  # number of vertical levels
params["dz0"] = 60  # height of first model level (m)
# if nz is None and for vgrid_method=0 only: specify maximum vertical grid spacing instead of nz;
# either float or "dx" to make it equal to dx
params["dzmax"] = 200
# method for creating vertical grid as defined in vertical_grid.py
# if None: do not change eta_levels
params["vgrid_method"] = 1

params["dt_f"] = 2  # time step (s), if None calculated as dt = 6 s/m *dx/1000; can be float
params["spec_hfx"] = None

params["input_sounding"] = "wrflux"  # name of input sounding to use (final name is then created: input_sounding_$name)
params["hm"] = 200  # mountain height (m)

# other standard namelist parameters
params["mp_physics"] = 0
params["bl_pbl_physics"] = 0
params["ra_lw_physics"] = 1
params["ra_sw_physics"] = 1
params["sf_surface_physics"] = 2

params["km_opt"] = 2
params["khdif"] = 0.
params["kvdif"] = 0.
params["use_theta_m"] = 1
params["mix_isotropic"] = 0
params["momentum_adv_opt"] = 1
params["moist_adv_opt"] = 1
params["scalar_adv_opt"] = 1
params["h_sca_adv_order"] = 5
params["v_sca_adv_order"] = 3
params["h_mom_adv_order"] = 5
params["v_mom_adv_order"] = 3

params["phi_adv_z"] = 2

# indices for output streams and their respective name and output interval (minutes, floats allowed)
# 0 is the standard output stream
params["output_streams"] = {24: ["meanout", 1.], 0: ["instout", 1.]}

params["restart_interval_m"] = 30  # restart interval (min)

params["min_nx_per_proc"] = 10  # 25, minimum number of grid points per processor
params["min_ny_per_proc"] = 10  # 25, minimum number of grid points per processor
