#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for submit_jobs.py
Test settings for automated tests.

@author: matze

"""
import os
import numpy as np
from collections import OrderedDict as odict
from run_wrf import misc_tools

#%%
'''Simulations settings'''
params = {} #parameter dict for params not used in param_grid

fpath = os.path.realpath(__file__)
fpath = fpath[:fpath.index("/fluxavg")]
wrf_dir_pre = fpath.split("/")[-1] #prefix for WRF build directory (_debug or _mpi are appended automatically)
ideal_case = "em_les" #idealized WRF case
runID = "pytest" #name for this simulation series

outpath = os.environ["wrf_res"] #WRF output path root
outdir = "test/" + runID #subdirectory for WRF output if not set in command line
run_path = os.environ["wrf_runs"] + "/" + runID #path where run directories of simulations will be created
build_path = os.environ["wrf_builds"] #path where different versions of the compiled WRF model code reside

# runID += "_new"
o = np.arange(2,7)
# names of parameter values for output filenames; either dictionaries or lists (not for composite parameters)
param_names = {"th" : ["thd", "thm", "thdm"],
               "th2" : ["thd", "thm"],
               "h_adv_order" : [2, 3],
               "v_adv_order" : [2, 3],
               "adv_order" : o,
               "bc" : ["open"]}

#Set additional namelist parameters (only active if they are not present in param_grid)
#any namelist parameters and some additional ones can be used


params["start_time"] = "2018-06-20_12:00:00" #format %Y-%m-%d_%H:%M:%S
params["end_time"] = "2018-06-20_13:00:00" #format %Y-%m-%d_%H:%M:%S

params["n_rep"] = 1 #number of repetitions for each configuration

#horizontal grid
params["dx"] = 500 #horizontal grid spacing (m)
params["lx"] = 5000 #minimum horizontal extent in east west (m)
params["ly"] = 5000 #minimum horizontal extent in north south (m)
#use minimum number of grid points set below:
use_min_gridpoints = False #"x", "y", True (for both) or False
params["min_gridpoints_x"] = 2 #minimum number of grid points in x direction (including boundary)
params["min_gridpoints_y"] = 2 #minimum number of grid points in y direction (including boundary)
#if use_min_gridpoints: force x and y extents to be multiples of lx and ly, respectively
force_domain_multiple = False #"x", "y", True (for both) or False

#control vertical grid creation (see vertical_grid.py for details on the different methods)
params["ztop"] = 3000 #top of domain (m)
params["zdamp"] = int(params["ztop"]/5) #depth of damping layer (m)
params["damp_opt"] = 0
params["nz"] = None #number of vertical levels
params["dz0"] = 60 #height of first model level (m)
params["dzmax"] = 300 #if nz is None and for dz_method=0 only: specify maximum vertical grid spacing instead of nz; either float or "dx" to make it equal to dx
params["dz_method"] = 3 #method for creating vertical grid as defined in vertical_grid.py

params["dt_f"] = 2  #time step (s), if None calculated as dt = 6 s/m *dx/1000; can be float
#minimum time between radiation calls (min); if radt is not specified: radt=max(radt_min, 10*dt)
params["radt_min"] = 1
params["spec_hfx"] = None

params["input_sounding"] = "unstable" #name of input sounding to use (final name is then created: input_sounding_$name)

#other standard namelist parameters
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


#indices for output streams and their respective name and output interval (minutes, floats allowed)
# 0 is the standard output stream
params["output_streams"] = {24: ["meanout", 10.], 0: ["instout", 10.] }

params["output_t_fluxes"] = 1
params["output_q_fluxes"] = 1
params["output_u_fluxes"] = 1
params["output_v_fluxes"] = 1
params["output_w_fluxes"] = 1
params["hesselberg_avg"] = True
params["output_dry_theta_fluxes"] = True


params["restart_interval_m"] = 30 #restart interval (min)

registries = ["Registry.EM_COMMON", "registry.hyb_coord", "registry.les", "registry.io_boilerplate"] #registries to look for default namelist parameters

# non-namelist parameters that will not be included in namelist file
del_args =   ["output_streams", "start_time", "end_time", "dz0", "dz_method", "min_gridpoints_x", "min_gridpoints_y", "lx", "ly", "spec_hfx", "input_sounding",
              "n_rep", "dt_f", "radt_min"]
#%%

nslots_dict = {} #set number of slots for each dx
min_n_per_proc = 30 #25, minimum number of grid points per processor
even_split = False #force equal split between processors
module_load = ""
#%%
'''Slot configurations and cluster settings'''
mail_address = "matthias.goebel@uibk.ac.at"
reduce_pool = True #reduce pool size to the actual uses number of slots; do not use if you do not want to share the node with others

pool_size = 4
cluster = False
max_nslotsy = None
max_nslotsx = None


