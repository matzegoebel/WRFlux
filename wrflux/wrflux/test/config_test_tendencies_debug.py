#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

@author: Matthias Göbel

"""
from wrflux.test.config_test_tendencies import *

from copy import deepcopy
params = deepcopy(params)

#%%
'''Simulations settings'''
runID = "pytest_debug" #name for this simulation series
params["end_time"] = "2018-06-20_12:02:00" #format %Y-%m-%d_%H:%M:%S
params["output_streams"] = {24: ["meanout", 1.], 0: ["instout", 1.] }

