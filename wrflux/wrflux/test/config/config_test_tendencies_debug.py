#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for launch_jobs.py
Test settings for automated tests with debug build.

@author: Matthias Göbel

"""
from wrflux.test.config.config_test_tendencies import *
from copy import deepcopy
params = deepcopy(params)


# %%

'''Simulations settings'''

# only short test run to check for invalid arithmetics,...
params["end_time"] = "2018-06-20_12:00:18"  # format %Y-%m-%d_%H:%M:%S
params["output_streams"] = {24: ["meanout", 0.1], 0: ["instout", 0.1]}
