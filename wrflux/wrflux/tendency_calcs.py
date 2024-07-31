#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:54:43 2020

Calculate the WRF budget terms for theta or qv under different assumptions.
Scatter and profile plots of the individual terms and sums.

@author: Matthias Göbel
"""
import datetime
import xarray as xr
from wrflux.test import testing
from wrflux import tools
import matplotlib.pyplot as plt
import sys
from pathlib import Path

try:
    # if mpi4py is not installed: no parallel processing possible
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.rank
    nproc = MPI.COMM_WORLD.size
    if nproc > 1:
        sys.stdout = open("p{}_tendency_calcs.log".format(rank), "w")
        sys.stderr = open("p{}_tendency_calcs.err".format(rank), "w")
except ImportError:
    rank = 0
    nproc = 1

xr.set_options(arithmetic_join="exact")
xr.set_options(keep_attrs=True)

start = datetime.datetime.now()

# %%settings
# path to WRF output

outpath_wrf = Path(__file__).parent / "example"
outpath = outpath_wrf / "postprocessed"
a = "out_d01_2018-06-20_12:00:00"
mean_file = "mean" + a
inst_file = "inst" + a

variables = ["t", "q", "u", "v", "w"]  # budget variables, q,t,u,v,w
avg_dims = ["y"]  # spatial averaging dimension, x and/or y
hor_avg = True  # average over avg_dims
t_avg = False  # average again over time
t_avg_interval = None  # size of the time averaging window

# select data before processing
# staggered dimension must have one gridpoint more than unstaggered

# pre_iloc = {"y" : slice(0,10), "y_stag" : slice(0,11)} #indices (do not use for Time!)
pre_iloc = None  # indices (do not use for Time!)
# pre_loc = {"Time": slice("2018-06-20 12:30:00", None)}  # labels
pre_loc = None

# skip postprocessing if data already exists
skip_exist = False

# Mapping from dimension "x" and/or "y" to chunk sizes to split the domain in tiles
# running script with mpirun enables parallel processing of tiles
if nproc > 1:
    chunks = {"x": 10}
else:
    chunks = None

# tests to perform
ignore = ["no_model_change", "adv_form", "Y=0"]
tests = [t for t in testing.all_tests if t not in ignore]

# %% set calculation methods

# available settings:
# cartesian: advective tendencies in Cartesian instead of native form
# adv_form : transform tendencies to advective form using the mass tendencies

# all budget calculation forms to apply as a list of str
# each item is a combination of setting strings from above separated by a space
budget_forms = ["", "cartesian", "cartesian adv_form"]

# %% calc tendencies

out = tools.calc_tendencies(
    variables,
    outpath_wrf,
    outpath,
    mean_file=mean_file,
    inst_file=inst_file,
    budget_forms=budget_forms,
    pre_iloc=pre_iloc,
    pre_loc=pre_loc,
    t_avg=t_avg,
    t_avg_interval=t_avg_interval,
    hor_avg=hor_avg,
    avg_dims=avg_dims,
    chunks=chunks,
    skip_exist=skip_exist,
    return_model_output=True,
)

# %% run tests
print("\n\n" + "#" * 50)
print("Run tests")
if rank == 0:
    datout, dat_inst, dat_mean = out
    kw = dict(
        avg_dims_error=[
            *avg_dims,
            "bottom_top",
            "Time",
        ],  # dimensions over which to calculate error norms
        plot=True,  # scatter plots for failed tests
        # plot_diff=True,  # plot difference between forcing and tendency against tendency
        discrete=True,  # discrete colormap
        # hue="x",
        ignore_missing_hue=True,
        savefig=True,
        figloc=outpath_wrf,
        # close = True # close figures directly
    )

    failed, err = testing.run_tests(
        datout, tests, dat_mean=dat_mean, dat_inst=dat_inst, hor_avg=hor_avg, chunks=chunks, **kw
    )

    if not (failed == "FAIL").any().any():
        print("\nAll tests passed")
    # %% plotting
    pdat = datout["t"]["tend"]["adv"].isel(x=15, Time=-1, dir=[0, 2, 3])
    if "y" in pdat.dims:
        pdat["z"] = pdat.z.mean("y")
        pdat = pdat.mean("y")
    pdat.name = "advective $\\theta$-tendency"
    pgrid = pdat.plot(hue="budget_form", row="dir", y="z", col="comp", sharex=False)
    plt.savefig(outpath_wrf / "tend_profile.pdf")

# %% elapsed time

print("\n\n" + "#" * 50)
time_diff = datetime.datetime.now() - start
time_diff = time_diff.total_seconds()

hours, remainder = divmod(time_diff, 3600)
minutes, seconds = divmod(remainder, 60)
print("Elapsed time: %s hours %s minutes %s seconds" % (int(hours), int(minutes), round(seconds)))
