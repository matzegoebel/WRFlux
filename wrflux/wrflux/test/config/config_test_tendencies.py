#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for launch_jobs.py
Test settings for automated tests with normal build.

@author: Matthias Göbel

"""
from wrflux.test.config.config_test_tendencies_base import *
from copy import deepcopy
params = deepcopy(params)

# %%
'''Simulations settings'''

params["end_time"] = "2018-06-20_13:00:00"  # format %Y-%m-%d_%H:%M:%S
params["output_streams"] = {24: ["meanout", 30.], 0: ["instout", 30.]}

params["output_t_fluxes"] = 1
params["output_q_fluxes"] = 1
params["output_u_fluxes"] = 1
params["output_v_fluxes"] = 1
params["output_w_fluxes"] = 1
params["output_t_fluxes_add"] = 3
params["output_q_fluxes_add"] = 3
params["output_u_fluxes_add"] = 3
params["output_v_fluxes_add"] = 3
params["output_w_fluxes_add"] = 3
params["hesselberg_avg"] = True
params["output_dry_theta_fluxes"] = True

params["iofields_filename"] = "IO_wdiag.txt"

# registries to look for default namelist parameters
registries = [*registries, "registry.wrflux"]