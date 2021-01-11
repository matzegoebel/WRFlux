#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:54:43 2020

Calculate the WRF budget terms for theta or qv under different assumptions.
Scatter and profile plots of the individual terms and sums.

@author: Matthias Göbel
"""

import os
import numpy as np
from wrflux import tools
from wrflux.test import testing
import xarray as xr
import datetime
from mpi4py import MPI
import netCDF4
rank = MPI.COMM_WORLD.rank
nproc = MPI.COMM_WORLD.size
import sys
if nproc > 1:
    sys.stdout = open('p{}_tendency_calcs.log'.format(rank), 'w')
    sys.stderr = open('p{}_tendency_calcs.err'.format(rank), 'w')

xr.set_options(arithmetic_join="exact")
xr.set_options(keep_attrs=True)

# os.environ["OMP_NUM_THREADS"]="4"
# os.environ["MKL_NUM_THREADS"]="4"
# os.environ["OPENBLAS_NUM_THREADS"]="4"

start = datetime.datetime.now()

#%%settings
outpath = "/home/c7071088/phd/results/wrf/test/pytest/pytest_0"
start_time="2018-06-20_12:00:00"

# avg_dim = ["x","y"] #spatial averaging dimension, x and/or y
avg_dims = ["y"]#spatial averaging dimension, x and/or y
hor_avg = True #average over avg_dim
t_avg = False #average over time
t_avg_interval = 4 #size of the time averaging window

#select data before processing
# pre_iloc = {"y" : slice(0,10), "y_stag" : slice(0,11)} #indices (do not use for Time!)
pre_iloc = None#indices (do not use for Time!)
pre_loc = {"Time" : slice("2018-06-20 12:30:00", None) } #labels
# pre_loc = None
variables = ["t", "q", "u", "v", "w"] #budget variables, q,t,u,v,w
# variables = ["u"]

save_output = True
skip_exist = False
chunks = {"x": 40}
# chunks = None


#%%set calculation methods

#all settings to test
budget_methods = [
            [[], "native"],
            ["cartesian", "correct"],
         ]

#%%

out = tools.calc_tendencies(variables, outpath,
                              budget_methods=budget_methods, start_time=start_time, pre_iloc=pre_iloc, pre_loc=pre_loc,
                              t_avg=t_avg, t_avg_interval=t_avg_interval, hor_avg=hor_avg, avg_dims=avg_dims,
                              chunks=chunks, save_output=save_output, skip_exist=skip_exist, return_model_output=True)

#%%
print("\n\n" + "#"*50)
print("Run tests")
if rank == 0:
    datout, dat_mean, dat_inst = out
    kw = dict(
        avg_dims_error = [*avg_dims, "bottom_top", "Time"], #dimensions over which to calculate error norms
        plot = True,
        # plot_diff = True, #plot difference between forcing and tendency against tendency
        discrete=True,
        # iloc={"x" : [*np.arange(9),*np.arange(-9,0)]},
        # iloc={"y" : [*np.arange(-9,0)]},
        # hue="x",
        s=40,
        ignore_missing_hue=True,
        # savefig = False,#TODOm
        # close = True
        )

    for var in variables:
        print("\nVariable: " + var)
        datout_v = datout[var]

        tend = datout_v["tend"].sel(comp="tendency")
        forcing = datout_v["tend"].sel(comp="forcing")
        failed, err = testing.test_budget(tend, forcing, thresh=0.999, **kw)
        testing.test_nan(datout_v)


print("\n\n" + "#"*50)
time_diff = datetime.datetime.now() - start
time_diff = time_diff.total_seconds()

hours, remainder = divmod(time_diff, 3600)
minutes, seconds = divmod(remainder, 60)
print("Elapsed time: %s hours %s minutes %s seconds" % (int(hours),int(minutes),round(seconds)))
