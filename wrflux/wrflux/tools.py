#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 12:53:04 2019

Part of WRFlux (https://github.com/matzegoebel/WRFlux)

Functions to calculate time-averaged tendencies from fluxes

@author: Matthias Göbel
"""
import xarray as xr
import logging
import netCDF4
import sys
import numpy as np
from wrflux.test import testing
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
from functools import partial
import itertools
try:
    # if mpi4py is not installed: no parallel processing possible
    from mpi4py.MPI import COMM_WORLD as comm
    rank = comm.rank
    nproc = comm.size
    if nproc > 1:
        sys.stdout = open('p{}_tendency_calcs.log'.format(rank), 'w')
        sys.stderr = open('p{}_tendency_calcs.err'.format(rank), 'w')
except ImportError:
    rank = 0
    nproc = 1
    comm = None

# logger for debugging messages
logger = logging.getLogger('l1')
logger.setLevel(logging.DEBUG)
# uncomment to enable debugging messages:
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# ch.setStream(sys.stdout)
# logger.addHandler(ch)

print = partial(print, flush=True)

xr.set_options(arithmetic_join="exact")
xr.set_options(keep_attrs=True)
DataArray = xr.core.dataarray.DataArray
Dataset = xr.core.dataset.Dataset

# %% constants

dim_dict = dict(x="U", y="V", bottom_top="W", z="W")
xy = ["x", "y"]
XY = ["X", "Y"]
XYZ = [*XY, "Z"]
uvw = ["u", "v", "w"]
units_dict = {"t": "K ", "q": "", **{v: "m s$^{-1}$" for v in uvw}}
units_dict_tend = {"t": "K s$^{-1}$", "q": "s$^{-1}$", **{v: "m s$^{-2}$" for v in uvw}}
units_dict_flux = {"t": "K m s$^{-1}$", "q": "m s$^{-1}$", **{v: "m$^{2}$ s$^{-2}$" for v in uvw}}
units_dict_tend_rho = {"t": "kg m$^{-3}$ K s$^{-1}$",
                       "q": "kg m$^{-3}$ s$^{-1}$", **{v: "kg m$^{-2}$ s$^{-2}$" for v in uvw}}
g = 9.81
rvovrd = 461.6 / 287.04
stagger_const = ["FNP", "FNM", "CF1", "CF2", "CF3", "CFN", "CFN1"]

# output datasets/dataarrays of postprocessing
outfiles = ["grid", "flux", "tend", "corr", "tend_mass"]
# attributes of WRF output variables to delete
del_attrs = ["MemoryOrder", "FieldType", "stagger", "coordinates"]

# available settings
budget_settings = ["cartesian", "adv_form", "dz_out_x", "dz_out_z", "force_2nd_adv"]
# abbreviations for settings
settings_short_names = {"2nd": "force_2nd_adv"}

all_variables = ["q", "t", "u", "v", "w"]


# %%open dataset

def open_dataset(file, del_attrs=True, fix_c=True, **kwargs):
    """
    Load file as xarray dataset.

    Parameters
    ----------
    file : str
        location of file to load.
    del_attrs : bool, optional
        Delete global attributes. The default is True.
    fix_c : bool, optional
        Assign time and space coordinates to dataset. The default is True.
    **kwargs :
        keyword arguments for xr.open_dataset

    Returns
    -------
    ds : xarray Dataset

    """
    kwargs["engine"] = "netcdf4"
    try:
        ds = xr.open_dataset(file, cache=False, **kwargs)
    except ValueError as e:
        if "unable to decode time" in e.args[0]:
            ds = xr.open_dataset(file, cache=False, decode_times=False, **kwargs)
        else:
            raise e
    ds.close()
    if fix_c:
        if "DX" in ds:
            dx, dy = ds.DX, ds.DY
        else:
            dx, dy = None, None
        ds = fix_coords(ds, dx=dx, dy=dy)

    if del_attrs:
        ds.attrs = {}

    ds.close()

    return ds


def fix_coords(data, dx=None, dy=None):
    """Assign time and space coordinates to dataset/dataarray."""
    # assign time coordinate
    if ("Time" in data.dims) and (type(data.Time.values[0]) == np.datetime64):
        pass
    elif "XTIME" in data.dims:
        data = data.rename(XTIME="Time")
    elif "Times" in data:
        time = data.Times.astype(str).values
        time = pd.DatetimeIndex([datetime.fromisoformat(str(t)) for t in time])
        data = data.assign_coords(Time=time)
    else:
        data = data.assign_coords(Time=data.XTIME)
    for v in ["XTIME", "Times"]:
        if v in data:
            data = data.drop(v)
    # assign x and y coordinates and rename dimensions
    for dim_old, res, dim_new in zip(["south_north", "west_east"], [dy, dx], ["y", "x"]):
        if res is None:
            continue
        for stag in [False, True]:
            if stag:
                dim_old = dim_old + "_stag"
                dim_new = dim_new + "_stag"
            if dim_old in data.dims:
                coord = np.arange(data.sizes[dim_old]) * res
                coord = coord - (coord[-1] + res) / 2
                coord = data[dim_old].assign_coords({dim_old: coord})[dim_old]
                coord = coord.assign_attrs({"units": "m"})
                data = data.assign_coords({dim_old: coord})
                data = data.rename({dim_old: dim_new})

    # assign vertical coordinate
    for v, dim in zip(["ZNU", "ZNW"], ["bottom_top", "bottom_top_stag"]):
        if (v in data) and (dim in data.dims):
            coord = data[v]
            if "Time" in coord.dims:
                coord = coord.isel(Time=0, drop=True)
            if "record" in coord.dims:
                if (coord.isel(record=0) != coord).any():
                    continue
                coord = coord.isel(record=0, drop=True)
            data = data.assign_coords({dim: coord})

    return data


# %%misc functions

def make_list(o):
    """Convert object to list if it is not already a tuple, list, dictionary, or array."""
    if type(o) not in [tuple, list, dict, np.ndarray]:
        o = [o]
    return o


def coarsen_avg(data, dim, interval, rho=None, mu=None, cyclic=None,
                stagger_kw=None, rho_weighted_vars=None, **avg_kwargs):
    """
    Coarsen and average dataset over the given dimension.

    If rho is given a density-weighted average is done.

    Parameters
    ----------
    data : dataset
        input dataset.
    dim : str
        dimension over which to apply the coarsening.
    interval : int
        averaging interval.
    rho : dataarray, optional
        If given, use to do density-weighted average. The default is None.
    mu : dataarray, optional
        If given, use to do mu-weighted average for fluxes. The default is None.
    cyclic : dict of booleans for xy or None, optional
        Defines which dimensions have periodic boundary conditions.
        If rho is given and needs to be staggered, use periodic boundary conditions
        to fill lateral boundary points. The default is None.
    stagger_kw : dict
        keyword arguments for staggering rho.
    rho_weighted_vars : list, optional
        List of variables to be density-weighted. Defaults to a prescribed list.

    Returns
    -------
    out : dataset
        coarsened dataset.

    """
    avg_kwargs = {dim: interval, "coord_func": {
        "Time": partial(select_ind, indeces=-1)}, "boundary": "trim"}

    if rho_weighted_vars is None:
        # define variables for density-weighting
        rho_weighted_vars = []
        exclude = ["CORR", "TEND", "RHOD_MEAN", "MUT_MEAN", "WW_MEAN", "_VAR"]
        for var in data.data_vars:
            if ("_MEAN" in var) and (var != "Z_MEAN") and all([e not in var for e in exclude]):
                rho_weighted_vars.append(var)

    if stagger_kw is None:
        stagger_kw = {}

    fluxes = ["F{}{}_ADV_MEAN".format(var.upper(), D) for D in XY for var in all_variables]
    var_stag = ["{}{}_MEAN".format(var.upper(), D) for D in XY for var in all_variables]
    mean_vel = ["U_MEAN", "V_MEAN"]
    mu_weighted_vars = fluxes + mean_vel + var_stag
    out = xr.Dataset()
    for var in data.data_vars:
        if (rho is not None) and (var in rho_weighted_vars):
            if var in mu_weighted_vars:
                # use mu instead of rho for weighting
                rho_i = build_mu(mu, data, full_levels="bottom_top_stag" in data[var].dims)
            else:
                rho_i = rho
            # density-weighted average
            rho_s = stagger_like(rho_i, data[var], cyclic=cyclic, **stagger_kw)
            rho_s_mean = rho_s.coarsen(**avg_kwargs).mean()
            out[var] = (data[var] * rho_s).coarsen(**avg_kwargs).mean() / rho_s_mean
        else:
            out[var] = data[var].coarsen(**avg_kwargs).mean()

    return out


def avg_xy(data, avg_dims, rho=None, cyclic=None, **stagger_const):
    """Average data over the given dimensions even
    if the actual present dimension has '_stag' added.
    If rho is given, do density-weighted averaging.
    Before averaging, cut right boundary points for periodic BC
    and both boundary points for non-periodic BC.

    Parameters
    ----------
    data : dataarray or dataset
        input data.
    avg_dims : list-like
        averaging dimensions.
    rho : dataarray, optional
        If given, use to do density-weighted average. The default is None.
    cyclic : dict of booleans for xy or None, optional
        Defines which dimensions have periodic boundary conditions.
        If rho is given and needs to be staggered, use periodic boundary conditions
        to fill lateral boundary points. The default is None.
    **stagger_kw :
        keyword arguments for staggering rho.

    Returns
    -------
    dataarray or dataset
        averaged data.

    """
    if type(data) == Dataset:
        out = xr.Dataset()
        for v in data.data_vars:
            out[v] = avg_xy(data[v], avg_dims, rho=rho, cyclic=cyclic, **stagger_const)
        return out

    if rho is not None:
        # prepare density-weighting
        rho_s = stagger_like(rho, data, cyclic=cyclic, **stagger_const)
        if cyclic is None:
            cyclic = {"x": False, "y": False}
        for d in rho_s.dims:
            if d not in data.dims:
                raise ValueError("Density in avg_xy contains dimension {} not present in data!".format(d))
            if (d not in rho.dims) and ("bottom_top" not in d) and (not cyclic[d[0]]):
                # avoid NaNs at the boundaries
                rho_s = rho_s.ffill(d)
                rho_s = rho_s.bfill(d)
        rho_s_mean = avg_xy(rho_s, avg_dims, cyclic=cyclic)

    # fix dimensions
    avg_dims_final = avg_dims.copy()
    for i, d in enumerate(avg_dims):
        ds = d + "_stag"
        if ds in data.dims:
            avg_dims_final.append(ds)
        if d not in data.dims:
            avg_dims_final.remove(d)

        # cut boundary points depending whether lateral BC are periodic or not
        if (cyclic is None) or (not cyclic[d]):
            for di in [d, ds]:
                data = loc_data(data, iloc={di: slice(1, -1)})
                if rho is not None:
                    rho_s = loc_data(rho_s, iloc={di: slice(1, -1)})
        elif ds in data.dims:
            #  for periodic BC only cut upper boundary in staggered dimension
            data = data[{ds: slice(0, -1)}]
            if rho is not None:
                rho_s = rho_s[{ds: slice(0, -1)}]

    # do (density-weighted) average
    if rho is None:
        return data.mean(avg_dims_final)
    else:
        return (data * rho_s).mean(avg_dims_final) / rho_s_mean


def warn_duplicate_dim(data, name=None):
    """Warn if dataarray or dataset contains both,
    the staggered and unstaggered version of any dimension.
    """
    if type(data) == Dataset:
        for v in data.data_vars:
            warn_duplicate_dim(data[v], name=v)
        return

    if name is None:
        name = data.name
    for d in data.dims:
        if d + "_stag" in data.dims:
            print("WARNING: datarray {0} contains both dimensions "
                  "{1} and {1}_stag".format(name, d))


def correct_dims_stag(loc, dat):
    """Correct keys of dictionary loc to fit to dimensions of dat.
    Add loc[key + "_stag"] = loc[key] (for staggered dimension) for
    every key in dictionary loc if that modified key is a dimension of
    dat and not already present in loc.
    Delete keys that are not dimensions of dat. Returns a copy of loc.

    Parameters
    ----------
    loc : dict
        input dictionary.
    dat : datarray or dataset
        reference data.

    Returns
    -------
    loc : dict
        modified dictionary.

    """
    loc_out = loc.copy()
    for d, val in loc.items():
        ds = d + "_stag"
        if (ds in dat.dims) and (ds not in loc):
            loc_out[ds] = val
        if d not in dat.dims:
            del loc_out[d]
    return loc_out


def correct_dims_stag_list(iterable, dat):
    """Correct items of iterable to fit to dimensions of dat.

    Add "_stag" to every item in iterable l, for which the unmodified item
    is not a dimension of dat but the modified item is. Delete items that are
    not dimensions of dat. Returns a copy.

    Parameters
    ----------
    iterable : iterable
        input iterable.
    dat : datarray or dataset
        reference data.

    Returns
    -------
    loc : list
        modified list.

    """
    l_out = []
    for i, d in enumerate(iterable):
        if d in dat.dims:
            l_out.append(d)
        else:
            if d + "_stag" in dat.dims:
                l_out.append(d + "_stag")
    return l_out


def loc_data(dat, loc=None, iloc=None, copy=True):
    """Apply label and integer-location based indexing with corrected dimensions."""
    if iloc is not None:
        iloc = correct_dims_stag(iloc, dat)
        dat = dat[iloc]
    if loc is not None:
        loc = correct_dims_stag(loc, dat)
        dat = dat.loc[loc]
    if copy:
        dat = dat.copy()

    return dat


# %%manipulate datasets
def select_ind(a, axis=0, indeces=0):
    """Select indeces along (possibly multiple) axis."""
    for axis in make_list(axis):
        a = a.take(indices=indeces, axis=axis)
    return a


def stagger_like(data, ref, rename=True, cyclic=None, ignore=None, **stagger_kw):
    """
    Stagger/Destagger all spatial dimensions to be consistent with reference data ref.

    Parameters
    ----------
    data : xarray dataarray or dataset
        input data.
    data : xarray dataarray
        reference data.
    rename : boolean, optional
        add "_stag" to dimension name. The default is True.
    cyclic : dict of booleans for xy or None, optional
        Defines which dimensions have periodic boundary conditions.
        Use periodic boundary conditions to fill lateral boundary points.
        The default is None.
    ignore : list, optional
        dimensions to ignore
    **stagger_kw : dict
        keyword arguments for staggering.

    Returns
    -------
    data : xarray dataarray or dataset
        output data.

    """
    if type(data) == Dataset:
        out = xr.Dataset()
        for v in data.data_vars:
            out[v] = stagger_like(data[v], ref, rename=rename,
                                  cyclic=cyclic, ignore=ignore, **stagger_kw)
        return out

    if ignore is None:
        ignore = []

    for d in data.dims:
        if (d.lower() != "time") and (d not in ref.dims) and (d not in ignore):
            if "stag" in d:
                data = destagger(data, d, ref[d[:d.index("_stag")]], rename=rename)
            else:
                ds = d + "_stag"
                if ds not in ref.coords:
                    continue
                if (cyclic is not None) and (d in cyclic):
                    cyc = cyclic[d]
                else:
                    cyc = False
                data = stagger(data, d, ref[ds], rename=rename, cyclic=cyc, **stagger_kw)

    return data


def stagger(data, dim, new_coord, FNM=0.5, FNP=0.5, rename=True, cyclic=False,
            fill_nearest=False, **interp_const):
    """
    Stagger WRF output data in given dimension by averaging neighbouring grid points.

    Parameters
    ----------
    data : xarray dataarray
        input data.
    dim : str
        staggering dimension.
    new_coord : array-like
        new coordinate to assign
    FNM : float or 1D array-like, optional
        upper weights for vertical staggering. The default is 0.5.
    FNP : float or 1D array-like, optional
        lower weights for vertical staggering. The default is 0.5.
    rename : boolean, optional
        add "_stag" to dimension name. The default is True.
    cyclic : bool, optional
        use periodic boundary conditions to fill lateral boundary points.
        The default is False.
    fill_nearest : bool, optional
        If cyclic=False, use nearest neighbour to fill missing values.
        The default is False.
    **interp_const : dict
        vertical extrapolation constants

    Returns
    -------
    data_stag : xarray dataarray
        staggered data.

    """
    if dim == "bottom_top":
        data_stag = data * FNM + data.shift({dim: 1}) * FNP
    else:
        data_stag = 0.5 * (data + data.roll({dim: 1}, roll_coords=False))

    data_stag = post_stagger(data_stag, dim, new_coord, rename=rename,
                             data=data, cyclic=cyclic, fill_nearest=fill_nearest,
                             **interp_const)

    return data_stag


def post_stagger(data_stag, dim, new_coord, rename=True, data=None, cyclic=False,
                 fill_nearest=False, **interp_const):
    """
    After staggering: rename dimension, assign new coordinate and fill boundary values.

    Parameters
    ----------
    data_stag : xarray dataarray
        staggered data.
    dim : str
        staggering dimension.
    new_coord : array-like
        new coordinate to assign
    rename : boolean, optional
        add "_stag" to dimension name. The default is True.
    data : xarray dataarray, optional
        unstaggered data for vertical extrapolation.
    cyclic : bool, optional
        use periodic boundary conditions to fill lateral boundary points.
        The default is False.
    fill_nearest : bool, optional
        If cyclic=False, use nearest neighbour to fill missing values.
        The default is False.
    **interp_const : dict
        vertical extrapolation constants

    Returns
    -------
    data_stag : xarray dataarray
        staggered data.
    """
    dim_s = dim
    if rename:
        dim_s = dim + "_stag"
        data_stag = data_stag.rename({dim: dim_s})

    data_stag[dim_s] = new_coord[:-1]
    data_stag = data_stag.reindex({dim_s: new_coord})

    c = new_coord

    # fill boundary values
    if dim == "bottom_top":
        if interp_const != {}:
            data_stag[{dim_s: 0}] = data[{dim: 0}]*interp_const["CF1"] + \
                data[{dim: 1}]*interp_const["CF2"] + data[{dim: 2}]*interp_const["CF3"]
            data_stag[{dim_s: -1}] = data[{dim: -1}]*interp_const["CFN"] + data[{dim: -2}]*interp_const["CFN1"]
    elif cyclic:
        # set second boundary point equal to first
        data_stag.loc[{dim_s: c[-1]}] = data_stag.loc[{dim_s: c[0]}]
    elif fill_nearest:
        # fill with neighbouring values
        data_stag.loc[{dim_s: c[0]}] = data_stag.loc[{dim_s: c[1]}]
        data_stag.loc[{dim_s: c[-1]}] = data_stag.loc[{dim_s: c[-2]}]
    else:
        # also set first boundary point to NaN
        data_stag.loc[{dim_s: c[0]}] = np.nan

    return data_stag


def destagger(data, dim, new_coord, rename=True):
    """
    Destagger WRF output data in given dimension by averaging neighbouring grid points.

    Parameters
    ----------
    data : xarray dataarray
        input data.
    dim : str
        destaggering dimension.
    new_coord : array-like
        new coordinate to assign
    rename : boolean, optional
        remove "_stag" from dimension name. The default is True.

    Returns
    -------
    data : xarray dataarray
        destaggered data.

    """
    data = 0.5 * (data + data.shift({dim: -1}))
    data = data.sel({dim: data[dim][:-1]})
    new_dim = dim
    if rename:
        new_dim = dim[:dim.index("_stag")]
        data = data.rename({dim: new_dim})

    data[new_dim] = new_coord

    return data


def diff(data, dim, new_coord=None, rename=True, cyclic=False):
    """
    Calculate first order differences along given dimension and assign new coordinates.

    Parameters
    ----------
    data : xarray dataarray
        input data.
    dim : str
        dimension over which to calculate the finite difference.
    new_coord : array-like, optional
        new coordinate to assign
    rename : boolean, optional
        remove/add "_stag" from dimension name. The default is True.
    cyclic : bool, optional
        if final (differenced) data is staggered: use periodic boundary conditions
        to fill lateral boundary points. The default is False.


    Returns
    -------
    out : xarray dataarray
        calculated differences.

    """
    if (dim in ["bottom_top", "z", "bottom_top_stag", "z_stag", "Time"]) or (not cyclic):
        # lower boundary value is nan
        data_s = data.shift({dim: 1})
    else:
        # lower boundary value is filled periodically
        data_s = data.roll({dim: 1}, roll_coords=False)

    out = data - data_s
    if "_stag" in dim:
        # if we go from staggered to unstaggered: lower boundary is not needed
        out = out.sel({dim: out[dim][1:]})
        # assign new coordinate
        new_dim = dim
        if rename and (dim != "Time"):
            new_dim = dim[:dim.index("_stag")]
            out = out.rename({dim: new_dim})
        if new_coord is None:
            new_coord = np.arange(len(out[new_dim]))
        out[new_dim] = new_coord
    else:
        if new_coord is None:
            new_coord = np.arange(len(out[dim]) + 1)
        # if we go from unstaggered to staggered: fill boundary values
        out = post_stagger(out, dim, new_coord, rename=rename, cyclic=cyclic)

    return out


def remove_deprecated_dims(ds):
    """Remove dimensions that do not occur in any of the variables of the given dataset."""
    # get all valid dims
    var_dims = []
    for v in ds.data_vars:
        var_dims.extend(ds[v].dims)
    # drop deprecated dims
    for d in ds.dims:
        if d not in var_dims:
            ds = ds.drop(d)
    return ds


# %%prepare tendencies


def load_data(outpath_wrf, inst_file, mean_file,
              pre_loc=None, pre_iloc=None, **kw):
    """Load WRF output data as xarray Datasets.

    Parameters
    ----------
    outpath_wrf : str
        Path to the WRF output directory.
    inst_file : str or path-like
        Name of the output file containing instantaneous data.
    mean_file : str
        Name of the output file containing time-averaged data.
    pre_loc : dict, optional
        Dictionary used for label based indexing of the input data before processing
        (e.g., {"Time" : slice("2018-06-20_12:00:00", None)}). The default is None.
    pre_iloc : dict, optional
        Dictionary used for integer-location based indexing of the input data before processing
        (e.g., {"x" : slice(10, -10)}). The default is None.
    **kw :
        Keyword arguments passed to load_data.

    Returns
    -------
    dat_mean : xarray Dataset
        WRF time-averaged output.
    dat_inst : xarray Dataset
        WRF instantaneous output.

    """
    kw["engine"] = "netcdf4"
    dat_inst = open_dataset(os.path.join(outpath_wrf, inst_file), del_attrs=False, **kw)
    dat_mean = open_dataset(os.path.join(outpath_wrf, mean_file), **kw)

    # select subset of data
    if pre_iloc is not None:
        if "Time" in pre_iloc:
            raise ValueError("Time axis should not be indexed with iloc, "
                             "as mean and inst output may have different frequencies!")
        dat_mean = dat_mean[pre_iloc]
        dat_inst = dat_inst[pre_iloc]
    if pre_loc is not None:
        dat_mean = dat_mean.loc[pre_loc]
        dat_inst = dat_inst.loc[pre_loc]
    if np.prod(list(dat_mean.sizes.values())) == 0:
        raise ValueError("At least one dimension is empy after indexing!")

    dims = ["record", "Time", "bottom_top", "bottom_top_stag", "y", "y_stag", "x", "x_stag"]
    dims_mean = [*[d for d in dims if d in dat_mean.dims],
                 *[d for d in dat_mean.dims if d not in dims]]
    dims_inst = [*[d for d in dims if d in dat_inst.dims],
                 *[d for d in dat_inst.dims if d not in dims]]
    dat_mean = dat_mean.transpose(*dims_mean)
    dat_inst = dat_inst.transpose(*dims_inst)

    return dat_mean, dat_inst


def load_postproc(outpath, var, cartesian, adv_form, hor_avg=False,
                  avg_dims=None, hor_avg_end=False):
    """
    Load already postprocessed data.

    Parameters
    ----------
    outpath : str or path-like
        Path to the postprocessed data.
    var : str
        Variable to load postprocessed output of.
    cartesian : bool
        If post-processed data includes calculations in Cartesian form,
        the file 'corr' will be loaded, as well.
    adv_form : bool
        Tendencies are in advective form
    hor_avg : bool, optional
        Load horizontally averaged output. The default is False.
    avg_dims : str or list of str, optional
        Averaging dimensions if hor_avg=True. The default is None.
    hor_avg_end : bool, optional
        Do horizontal average at end of processing step instead of inside.
        This affects the mean and turbulent components. The default is False.

    Returns
    -------
    datout : dict
        Postprocessed output as nested dictionary.

    """
    if hor_avg or hor_avg_end:
        avg = "_avg_" + "".join(avg_dims)
        if hor_avg_end:
            avg += "_end"
    else:
        avg = ""
    datout = {}
    outpath = Path(outpath) / var.upper()
    for f in outfiles:
        if (f == "corr") and (not cartesian):
            continue
        if (f == "tend_mass") and (var != "t"):
            continue
        file = outpath / (f + avg + ".nc")
        try:
            datout[f] = xr.open_dataarray(file, cache=False, engine="netcdf4")
        except ValueError:
            datout[f] = xr.open_dataset(file, cache=False, engine="netcdf4")
        datout[f].close()

    return datout


def get_budget_method(budget_method):
    """Build settings dictionary from str. Replace abbreviations."""
    if budget_method == "":
        # if no settings are given: calculate native WRF tendencies
        budget_method = "native"
        budget_method_list = []
    else:
        budget_method = budget_method.strip()
        budget_method_list = budget_method.split(" ")

    for i, key in enumerate(budget_method_list):
        if key in settings_short_names:
            budget_method_list[i] = settings_short_names[key]

    config = {}
    undefined = [key for key in budget_method_list if key not in budget_settings]
    if len(undefined) > 0:
        raise ValueError("Undefined keys: {}".format(", ".join(undefined)))
    for k in budget_settings:
        if k in budget_method_list:
            config[k] = True
        else:
            config[k] = False
    return config, budget_method


def prepare(dat_mean, dat_inst, variables, cyclic=None,
            t_avg=False, t_avg_interval=None, hor_avg=False, avg_dims=None):
    """Prepare tendency calculations.

    Define grid variables add vertical coordinates to WRF output data.
    and average over time if desired. If time-averaging is performed and the
    original output contains all time steps, turbulent fluxes are
    computed explicitly.


    Parameters
    ----------
    dat_mean : xarray Dataset
        WRF time-averaged output.
    dat_inst : xarray Dataset
        WRF instantaneous output.
    variables : list of str
        List of variables to process.
    cyclic : dict of booleans for xy or None, optional
        Defines which dimensions have periodic boundary conditions.
        Use periodic boundary conditions to fill lateral boundary points.
        The default is None.
    t_avg : bool, optional
        Average WRF output again over time. The default is False.
    t_avg_interval : integer, optional
        Interval for time averaging (number of output time steps) if t_avg=True.
        The default is None.
    hor_avg : bool, optional
        Average horizontally. The default is False.
    avg_dims : str or list of str, optional
        Averaging dimensions if hor_avg=True. The default is None.

    Returns
    -------
    dat_mean : xarray Dataset
        Modified WRF time-averaged output.
    dat_inst : xarray Dataset
        Modified WRF instantaneous output.
    grid : xarray Dataset
        Variables related to the model grid.
    attrs : dict
        Global attributes of WRF output.

    """
    print("Prepare data")
    attrs = dat_inst.attrs
    dat_inst.attrs = {}

    # strip first time as dat_inst needs to be one time stamp longer
    dat_mean = dat_mean.sel(Time=dat_mean.Time[1:])
    if len(dat_mean.Time) == 0:
        raise ValueError("dat_mean is empty! Needs to contain at least two timesteps initially!")

    # computational grid
    grid = dat_inst[["ZNU", "ZNW", "DNW", "DN", "C1H", "C2H",
                     "C1F", "C2F", *stagger_const]].isel(Time=0, drop=True)
    if "record" in grid.dims:
        for v in ["ZNU", "ZNW"]:
            if (grid[v].isel(record=0) != grid[v]).any():
                raise ValueError("If record dim is present in input data, eta levels "
                                 "needs to be identical for all records!")
            else:
                grid[v] = grid[v].isel(record=0, drop=True)

    dn = grid["DN"].rename(bottom_top="bottom_top_stag").assign_coords(bottom_top_stag=grid["bottom_top_stag"][:-1])
    grid["DN"] = dn.reindex(bottom_top_stag=grid["bottom_top_stag"])
    grid["DX"] = attrs["DX"]
    grid["DY"] = attrs["DY"]
    dat_mean = dat_mean.assign_coords(bottom_top=grid["ZNU"], bottom_top_stag=grid["ZNW"])
    dat_inst = dat_inst.assign_coords(bottom_top=grid["ZNU"], bottom_top_stag=grid["ZNW"])
    grid = grid.assign_coords(bottom_top=grid["ZNU"], bottom_top_stag=grid["ZNW"])

    dat_mean = dat_mean.rename(ZWIND_MEAN="W_MEAN")
    rhod = stagger(dat_mean["RHOD_MEAN"], "bottom_top",
                   dat_mean["bottom_top_stag"], **grid[stagger_const])
    dat_mean["OMZN_MEAN"] = dat_mean["WW_MEAN"] / (-g * rhod)

    if t_avg:
        inst = dat_mean.copy()
        print("Average dat_mean over {} output steps".format(t_avg_interval))
        dat_mean = coarsen_avg(dat_mean, dim="Time", interval=t_avg_interval,
                               rho=inst["RHOD_MEAN"], mu=inst["MUT_MEAN"],
                               cyclic=cyclic, stagger_kw=grid[stagger_const])

        # compute resolved turbulent fluxes explicitly if output contains all timesteps
        dt_out = float(inst.Time[1] - inst.Time[0]) / 1e9
        if round(dt_out) == attrs["DT"]:
            attrs["trb_exp"] = 1
            print("Compute turbulent fluxes explicitly")
            testing.trb_fluxes(dat_mean, inst, variables, grid, t_avg_interval,
                               cyclic=cyclic, hor_avg=hor_avg, avg_dims=avg_dims)

    # restrict instantaneous data to time steps contained in averaged data
    if attrs["AVG_INTERVAL"] == -1:
        # averaging interval is output interval of dat_mean
        avg_interval = (dat_mean.Time[1:].values - dat_mean.Time[:-1].values).mean()
        attrs["AVG_INTERVAL"] = int(avg_interval) / 1e9
    else:
        avg_interval = np.timedelta64(int(attrs["AVG_INTERVAL"] * 1e9))

    # select start and end time of averaging intervals
    Times = []
    for time in dat_mean.Time.values:
        Times.extend([time - avg_interval, time])

    try:
        dat_inst = dat_inst.sel(Time=Times)
    except KeyError:
        times_str = [datetime.utcfromtimestamp(time.tolist()/1e9).isoformat() for time in Times]
        raise KeyError("Not all necessary time steps included in instantaneous data: "
                       "need start and end times of all averaging intervals. "
                       "Needed times: {}".format(times_str))

    for v in dat_inst.coords:
        if ("XLAT" in v) or ("XLONG" in v):
            dat_inst = dat_inst.drop(v)

    return dat_mean, dat_inst, grid, attrs


def build_mu(mut, grid, full_levels=False):
    """Build 3D mu on full or half-levels from 2D mu and grid constants."""
    if full_levels:
        mu = grid["C1F"] * mut + grid["C2F"]
    else:
        mu = grid["C1H"] * mut + grid["C2H"]
    return mu

# %% WRF tendencies


def calc_tend_sources(dat_mean, dat_inst, var, grid, cyclic, attrs, hor_avg=False, avg_dims=None):
    """Calculate and sum up tendencies from SGS fluxes and other source terms except advection and
    calculate some helper variables specific to the variable being processed.

    Parameters
    ----------
    dat_mean : xarray Dataset
        WRF time-averaged output.
    dat_inst : xarray Dataset
        WRF instantaneous output.
    var : str
        Variable to process.
    grid : xarray Dataset
        Variables related to the model grid.
    cyclic : dict of booleans for xy
        Defines which dimensions have periodic boundary conditions.
        Use periodic boundary conditions to fill lateral boundary points.
    attrs : dict
        Global attributes of WRF output.
    hor_avg : bool, optional
        Average horizontally. The default is False.
    avg_dims : str or list of str, optional
        Averaging dimensions if hor_avg=True. The default is None.

    Returns
    -------
    dat_mean : xarray Dataset
        modified WRF time-averaged output.
    dat_inst : xarray Dataset
        modified WRF instantaneous output.
    sgs : xarray DataArray
        SGS tendencies calculated from SGS fluxes in all three directions.
    sgsflux : xarray Dataset
        SGS fluxes in all three directions.
    sources : xarray Dataset
        All source terms for variable except advection.
    sources_sum : xarray DataArray
        Sum of source terms.
    grid : xarray Dataset
        Modified grid data.
    dim_stag : str
        Staggering dimension of variable.
    mapfac : xarray Dataset
        Map-scale factor variables.

    """
    print("\nPrepare tendency calculations for {}".format(var.upper()))

    VAR = var.upper()
    dim_stag = None  # for momentum: staggering dimension
    if var == "u":
        dim_stag = "x"
    elif var == "v":
        dim_stag = "y"
    elif var == "w":
        dim_stag = "bottom_top"

    # mapscale factors
    if var in ["u", "v"]:
        mapfac_type = VAR
    else:
        mapfac_type = "M"
    mapfac_vnames = ["MAPFAC_{}X".format(mapfac_type), "MAPFAC_{}Y".format(mapfac_type)]
    mapfac = dat_inst[mapfac_vnames].isel(Time=0, drop=True)
    mapfac = mapfac.rename(dict(zip(mapfac_vnames, XY)))

    # map-scale factors for fluxes
    for d, m in zip(XY, ["UY", "VX"]):
        mf = dat_inst["MAPFAC_" + m].isel(Time=0, drop=True)
        flx = dat_mean["F{}{}_ADV_MEAN".format(var[0].upper(), d)]
        mapfac["F" + d] = stagger_like(mf, flx, cyclic=cyclic)

    # get missing sgs flux
    dat_mean["FUY_SGS_MEAN"] = dat_mean["FVX_SGS_MEAN"]

    # density and dry air mass
    mut = dat_inst["MU"] + dat_inst["MUB"]
    mu = build_mu(mut, grid)
    dat_inst["MU_STAG"] = mu
    grid["MU_STAG_MEAN"] = build_mu(dat_mean["MUT_MEAN"], grid)
    rhodm = dat_mean["RHOD_MEAN"]
    if var in uvw:
        rhodm = stagger(rhodm, dim_stag, dat_inst[dim_stag + "_stag"],
                        cyclic=cyclic[dim_stag], **grid[stagger_const])
        if var == "w":
            dat_inst["MU_STAG"] = build_mu(mut, grid, full_levels=True)
            grid["MU_STAG_MEAN"] = build_mu(dat_mean["MUT_MEAN"], grid, full_levels=True)
        else:
            dat_inst["MU_STAG"] = stagger(dat_inst["MU_STAG"], dim_stag,
                                          dat_inst[dim_stag + "_stag"], cyclic=cyclic[dim_stag])
            grid["MU_STAG_MEAN"] = stagger(grid["MU_STAG_MEAN"], dim_stag,
                                           dat_mean[dim_stag + "_stag"], cyclic=cyclic[dim_stag])

    ref = rhodm
    if hor_avg:
        rhodm = avg_xy(rhodm, avg_dims, cyclic=cyclic)
    grid["RHOD_STAG_MEAN"] = rhodm

    # derivative of z wrt x,y,t
    dzdd = xr.Dataset()
    for d in xy:
        D = d.upper()
        dzdd[D] = diff(dat_mean["Z_MEAN"], d, dat_mean[d + "_stag"],
                       cyclic=cyclic[d]) / grid["D" + D]

    zw_inst = (dat_inst["PH"] + dat_inst["PHB"]) / g
    dt = attrs["AVG_INTERVAL"]
    # only keep end points of averaging intervals
    dzdd["T"] = zw_inst.diff("Time").isel(Time=slice(None, None, 2)) / dt
    dzdd = stagger_like(dzdd, ref, ignore=["bottom_top_stag"], cyclic=cyclic)
    dzdd = remove_deprecated_dims(dzdd)
    grid["dzdd"] = dzdd.to_array("dir").assign_attrs(
        description="derivative of instantaneous geopotential height "
                    "with respect to x, y, and t")
    for d in XY:
        dz = dat_mean["DPH_{}_MEAN".format(d)] / g
        dzdt_d = stagger_like(dz, ref, ignore=["bottom_top_stag"], cyclic=cyclic)
        if hor_avg:
            dzdt_d = avg_xy(dzdt_d, avg_dims, rho=dat_mean["RHOD_MEAN"],
                            cyclic=cyclic, **grid[stagger_const])

        desc = dzdt_d.description.replace("tendency", "height tendency")
        grid["dzdt_{}".format(d.lower())] = dzdt_d.assign_attrs(description=desc,
                                                                units="m s-1")
    # instantaneous density
    rhod = - 1 / diff(g * zw_inst, "bottom_top_stag", dat_inst.bottom_top) * grid["DNW"] * mu
    # (de)stagger rhod to grid of var
    dat_inst["RHOD_STAG"] = stagger_like(rhod, ref, cyclic=cyclic, **grid[stagger_const])

    # height
    grid["ZW"] = dat_mean["Z_MEAN"]
    grid["Z_STAG"] = stagger_like(dat_mean["Z_MEAN"], ref, cyclic=cyclic, **grid[stagger_const])

    # additional sources
    print("Compute SGS and additional tendencies")
    sources = xr.Dataset()
    cu = (attrs["CU_PHYSICS"] > 0) or (attrs["SHCU_PHYSICS"] > 0)
    if var == "t":
        if attrs["MP_PHYSICS"] > 0:
            sources["mp"] = dat_mean["T_TEND_MP_MEAN"]
        if attrs["RA_LW_PHYSICS"] > 0:
            sources["rad_lw"] = dat_mean["T_TEND_RADLW_MEAN"]
        if attrs["RA_SW_PHYSICS"] > 0:
            sources["rad_sw"] = dat_mean["T_TEND_RADSW_MEAN"]
        if cu:
            sources["cu"] = dat_mean["T_TEND_CU_MEAN"]
    elif var == "q":
        if attrs["MP_PHYSICS"] > 0:
            sources["mp"] = dat_mean["Q_TEND_MP_MEAN"]
        if cu:
            sources["cu"] = dat_mean["Q_TEND_CU_MEAN"]
    else:
        # pressure gradient, buoyancy, coriolis and curvature
        sources["pg"] = dat_mean["{}_TEND_PG_MEAN".format(VAR)]
        cor = "{}_TEND_COR_CURV_MEAN".format(VAR)
        if cor in dat_mean:
            sources["cor_curv"] = dat_mean[cor]
        if (VAR != "W") and cu:
            cu = dat_mean["{}_TEND_CU_MEAN".format(VAR)]
            # convection tendency is on mass levels
            sources["cu"] = stagger_like(cu, ref, cyclic=cyclic)
    damp = "{}_TEND_DAMP_MEAN".format(VAR)
    if damp in dat_mean:
        sources["damp"] = dat_mean[damp]

    # calculate tendencies from sgs fluxes
    sgs = []
    sgsflux = []
    for cartesian, label in zip([True, False], ["cartesian", "native"]):
        sgs_i, sgsflux_i = sgs_tendency(dat_mean, VAR, grid, cyclic,
                                        cartesian=cartesian, mapfac=mapfac)
        sgs.append(sgs_i.expand_dims(ID=[label]))
        sgsflux.append(sgsflux_i.expand_dims(ID=[label]))
    sgs = xr.concat(sgs, dim="ID")
    sgsflux = xr.concat(sgsflux, dim="ID")

    if hor_avg:
        sources = avg_xy(sources, avg_dims, cyclic=cyclic)
        sgs = avg_xy(sgs, avg_dims, cyclic=cyclic)
        sgsflux = avg_xy(sgsflux, avg_dims, cyclic=cyclic)
        grid["dzdd"] = avg_xy(grid["dzdd"], avg_dims, rho=dat_mean["RHOD_MEAN"],
                              cyclic=cyclic, **grid[stagger_const])
        grid = avg_xy(grid, avg_dims, cyclic=cyclic)

    sgs_sum = sgs.sum("dir", skipna=False)
    if len(sources.data_vars) > 0:
        sources_sum = sources.to_array("comp").sum("comp")
    else:
        sources_sum = 0
    sources_sum = sources_sum + sgs_sum.sel(ID="cartesian", drop=True)

    return dat_mean, dat_inst, sgs, sgsflux, sources, sources_sum, grid, dim_stag, mapfac


def sgs_tendency(dat_mean, VAR, grid, cyclic, cartesian=False, mapfac=None):
    """Calculate tendencies from SGS fluxes.

    Parameters
    ----------
    dat_mean : xarray Dataset
        WRF time-averaged output.
    VAR : str
        variable to process.
    grid : xarray Dataset
        variables related to the model grid.
    cyclic : dict of booleans for xy
        Defines which dimensions have periodic boundary conditions.
        Use periodic boundary conditions to fill lateral boundary points.
    cartesian : bool, optional
        Calculate the tendencies in Cartesian form. The correction terms
        are then applied to the horizontal flux derivatives.
        Otherwise they are applied to the vertical derivative.
        The default is False.
    mapfac : xarray Dataset, optional
        Map-scale factor variables. The default is None.

    Returns
    -------
    sgs : xarray DataArray
        SGS tendencies calculated from SGS fluxes in all three directions.
    sgsflux : xarray Dataset
        SGS fluxes in all three directions.

    """
    sgs = xr.Dataset()
    sgsflux = xr.Dataset()
    if VAR == "W":
        d3s = "bottom_top"
        d3 = "bottom_top_stag"
        vcoord = grid["ZNW"]
        dn = grid["DN"]
    else:
        d3 = "bottom_top"
        d3s = "bottom_top_stag"
        vcoord = grid["ZNU"]
        dn = grid["DNW"]
    fz = dat_mean["F{}Z_SGS_MEAN".format(VAR)]
    fz = fz * stagger_like(dat_mean["RHOD_MEAN"], fz, cyclic=cyclic, **grid[stagger_const])
    for d, v in zip(xy, ["U", "V"]):
        # horizontal derivatives and Cartesian corrections
        D = d.upper()
        if mapfac is None:
            mf = 1
        else:
            mf = mapfac[D]
        fd = dat_mean["F{}{}_SGS_MEAN".format(VAR, D)]
        sgsflux[D] = fd
        fd = fd * stagger_like(dat_mean["RHOD_MEAN"], fd, cyclic=cyclic, **grid[stagger_const])
        cyc = cyclic[d]
        # (de)stagger flux horizontally
        if d in fd.dims:
            # for momentum variances
            ds = d
            d = ds + "_stag"
            flux8v = stagger(fd, ds, new_coord=dat_mean[d], cyclic=cyc)
        else:
            ds = d + "_stag"
            flux8v = destagger(fd, ds, new_coord=dat_mean[d])

        # flux derivative
        dx = grid["D" + D]
        sgs[D] = -diff(fd, ds, new_coord=dat_mean[d], cyclic=cyc) / dx * mf / grid["RHOD_STAG_MEAN"]

        # (de)stagger flux vertically
        if VAR == "W":
            flux8z = destagger(flux8v, d3, grid["ZNU"])
            mf = mapfac["Y"]
        else:
            flux8z = stagger(flux8v, d3, grid["ZNW"], **grid[stagger_const])
            flux8z[{d3s: [0, -1]}] = 0

        # build Cartesian correction
        corr_sgs = diff(flux8z, d3s, new_coord=vcoord) / dn
        corr_sgs = corr_sgs * mf * stagger_like(grid["dzdd"].loc[D], corr_sgs, cyclic=cyclic, **grid[stagger_const])
        corr_sgs_m = corr_sgs / grid["MU_STAG_MEAN"] * (-g)
        if cartesian:
            sgs[D] = sgs[D] + corr_sgs_m
        else:
            # vertical flux is in Cartesian coordinate system
            # -> transform to native by vertical integration of correction
            dfz = (corr_sgs * dn)[{d3: slice(None, None, -1)}].cumsum(d3)[{d3: slice(None, None, -1)}]
            if VAR == "W":
                fz = fz + dfz[{d3: slice(1, None)}].values
            else:
                # top flux is zero: only set values below
                fz[{d3s: slice(None, -1)}] = fz[{d3s: slice(None, -1)}] + dfz.values
    # vertical flux derivative
    sgsflux["Z"] = fz
    sgs["Z"] = -diff(fz, d3s, new_coord=vcoord) / dn / grid["MU_STAG_MEAN"] * (-g)
    sgs = sgs[XYZ]
    sgs = sgs.to_array("dir")
    if VAR == "W":
        # no sgs tendency at surface and domain top
        sgs[{d3: [0, -1]}] = 0

    return sgs, sgsflux


def adv_tend(dat_mean, dat_inst, VAR, grid, mapfac, cyclic, attrs,
             hor_avg=False, avg_dims=None, calc_mass=False, cartesian=True,
             force_2nd_adv=False,  dz_out_x=False, dz_out_z=False):
    """Compute advective tendencies decomposed into mean and resolved turbulence.

    Also return Cartesian corrections, but do not apply them yet.

    Parameters
    ----------
    dat_mean : xarray Dataset
        WRF time-averaged output.
    dat_inst : xarray Dataset
        WRF instantaneous output.
    VAR : str
        Variable to process.
    grid : xarray Dataset
        Variables related to the model grid.
    mapfac : xarray Dataset
        Map-scale factor variables.
    cyclic : dict of booleans for xy
        Defines which dimensions have periodic boundary conditions.
        Use periodic boundary conditions to fill lateral boundary points.
    attrs : dict
        Global attributes of WRF output.
    hor_avg : bool, optional
        Average horizontally. The default is False.
    avg_dims : str or list of str, optional
        Averaging dimensions if hor_avg=True. The default is None.
    calc_mass : bool, optional
        Calculate mass tendencies. The default is False.
    cartesian : bool, optional
        Calculate Cartesian tendencies. The default is False.
    force_2nd_adv : bool, optional
        Use 2nd-order fluxes. The default is False.
    dz_out_x : bool, optional
        In Cartesian correction terms: take derivatives of z out of vertical derivative.
        Stagger horizontal flux to the grid of the vertical flux.
        The default is False.
    dz_out_z : bool, optional
        In Cartesian correction terms: take derivatives of z out of vertical derivative.
        Use variable correctly staggered in the vertical (depending on advection order).
        The default is True.

    Returns
    -------
    flux : xarray Dataset
        Decomposed advective fluxes in all three directions.
    adv : xarray Dataarray
        Decomposed advective tendencies from all three directions.
    vmean : xarray Dataset
        Mean velocities.
    var_stag : xarray Dataset
        Averaged variable VAR staggered to the grid of the three spatial fluxes.
    corr_flx : xarray Dataarray
        Cartesian correction fluxes.
    tend_mass : xarray DataArray
        Components of continuity equation.z

    """
    print("Compute resolved tendencies")

    # get appropriately staggered variables
    var_stag = xr.Dataset()
    fluxnames = ["F{}{}_ADV_MEAN".format(VAR, d) for d in XYZ]
    if force_2nd_adv:
        fluxnames = [fn + "_2ND" for fn in fluxnames]
        for d, f in zip(XYZ, fluxnames):
            var_stag[d] = stagger_like(dat_mean["{}_MEAN".format(VAR)],
                                       dat_mean[f], cyclic=cyclic, **grid[stagger_const])
        if VAR == "T":
            if attrs["USE_THETA_M"] and (not attrs["OUTPUT_DRY_THETA_FLUXES"]):
                raise ValueError(
                    "Averaged moist potential temperature not available to build "
                    "mean 2nd-order fluxes! (use_theta_m=1 and output_dry_theta_fluxes=0)")
    else:
        for d in XYZ:
            var_stag[d] = dat_mean["{}{}_MEAN".format(VAR, d)]

    # mean velocities
    if cartesian:
        w = dat_mean["WD_MEAN"]
    else:
        w = dat_mean["OMZN_MEAN"]
    vmean = xr.Dataset({"X": dat_mean["U_MEAN"], "Y": dat_mean["V_MEAN"], "Z": w})

    if not all([f in dat_mean for f in fluxnames]):
        raise ValueError("Fluxes not available!")
    tot_flux = dat_mean[fluxnames]
    tot_flux = tot_flux.rename(dict(zip(fluxnames, XYZ)))
    rhod8z = stagger_like(dat_mean["RHOD_MEAN"], tot_flux["Z"],
                          cyclic=cyclic, **grid[stagger_const])

    # Standard Cartesian corrections
    corr_flx = ["F{}X_CORR".format(VAR), "F{}Y_CORR".format(VAR), "CORR_D{}DT".format(VAR)]
    if force_2nd_adv:
        corr_flx = [corri + "_2ND" for corri in corr_flx]
    corr_flx = dat_mean[corr_flx]
    corr_flx = corr_flx.to_array("dir")
    corr_flx["dir"] = ["X", "Y", "T"]

    if not cartesian:
        # corrections were already included in vertical flux: remove them
        tot_flux["Z"] = tot_flux["Z"] - (corr_flx.loc["X"] + corr_flx.loc["Y"] + corr_flx.loc["T"]) / rhod8z
        tot_flux = tot_flux.drop("dir")

    # Use alternative corrections
    dz_out = False
    if dz_out_x or dz_out_z:
        dz_out = True
        if dz_out_z:
            corr_flx.loc["X"] = dat_mean["F{}X_CORR_DZOUT".format(VAR)]
            corr_flx.loc["Y"] = dat_mean["F{}Y_CORR_DZOUT".format(VAR)]
        else:
            corr_new = tot_flux[XY]
            corr_new = rhod8z * stagger_like(corr_new, rhod8z, cyclic=cyclic, **grid[stagger_const])
            corr_flx.loc["X"] = corr_new["X"]
            corr_flx.loc["Y"] = corr_new["Y"]
        corr_t = rhod8z * dat_mean[VAR + "Z_MEAN"]
        corr_flx.loc["T"] = corr_t

    #  mean advective fluxes
    mean_flux = xr.Dataset()
    mass_flux = vmean.copy()
    for d in XYZ:
        if d in ["X", "Y"]:
            rho = build_mu(dat_mean["MUT_MEAN"], grid, full_levels="bottom_top_stag" in vmean[d].dims)
        else:
            rho = dat_mean["RHOD_MEAN"]

        rho_stag = stagger_like(rho, ref=vmean[d], cyclic=cyclic, **grid[stagger_const], fill_nearest=True)
        vel_stag = stagger_like(vmean[d] * rho_stag, ref=var_stag[d], cyclic=cyclic, **grid[stagger_const])
        if (VAR == "W") and (d in XY):
            vel_stag[{"bottom_top_stag": 0}] = 0
        if calc_mass:
            rho_stag = stagger_like(rho, ref=vel_stag, cyclic=cyclic, **grid[stagger_const])
            mass_flux[d] = vel_stag / rho_stag
        rho_m = rho
        if hor_avg and (d.lower() not in avg_dims):  # TODO: use normal rho for var_stag...?
            # average horizontally, but only if current dimension is not averaging dimension
            var_stag[d] = avg_xy(var_stag[d], avg_dims, rho=rho, cyclic=cyclic, **grid[stagger_const])
            vmean[d] = avg_xy(vmean[d], avg_dims, rho=rho, cyclic=cyclic, **grid[stagger_const])
            vel_stag = avg_xy(vel_stag, avg_dims, rho=rho, cyclic=cyclic, **grid[stagger_const])
            rho_m = avg_xy(rho, avg_dims, cyclic=cyclic, **grid[stagger_const])

        rho_stag = stagger_like(rho_m, ref=vel_stag, cyclic=cyclic, **grid[stagger_const])
        mean_flux[d] = var_stag[d] * vel_stag / rho_stag
        if hor_avg and (d.lower() in avg_dims):
            # now average over dimension that has been skipped before
            var_stag[d] = avg_xy(var_stag[d], avg_dims, rho=rho, cyclic=cyclic, **grid[stagger_const])
            vmean[d] = avg_xy(vmean[d], avg_dims, rho=rho, cyclic=cyclic, **grid[stagger_const])

    fluxes = {"adv_r": tot_flux, "mean": mean_flux}
    if calc_mass:
        # mass flux for advection of constant base state
        fluxes["mass"] = mass_flux
    if ("trb_exp" in attrs) and (attrs["trb_exp"] == 1):
        # use explicit resolved turbulent fluxes if present
        trb_flux = xr.Dataset()
        if cartesian:
            vels = ["U", "V", "W"]
        else:
            vels = ["U", "V", "OMZN"]
        for d, vel in zip(XYZ, vels):
            trb_flux[d] = dat_mean["F{}{}_TRB_MEAN".format(VAR, vel)]
        fluxes["trb_r"] = trb_flux
        trb_exp = True
    else:
        trb_exp = False

    adv = {}
    # advective tendency from fluxes
    for comp, flux in fluxes.items():
        adv_i = xr.Dataset()
        # horizontal fluxes first
        for d in xy:
            D = d.upper()
            cyc = cyclic[d]
            if d in flux[D].dims:
                ds = d
                d = d + "_stag"
            else:
                ds = d + "_stag"

            # determine correct mapscale factors and density to multiply with flux
            mf_flx = mapfac["F" + D]
            if dz_out:
                # only need density not dry air mass
                fac = dat_mean["RHOD_MEAN"]
            else:
                fac = dat_mean["MUT_MEAN"]
            if (comp in ["mean", "trb_r"]) and hor_avg and (d not in avg_dims):
                fac = avg_xy(fac, avg_dims, cyclic=cyclic)
            if not dz_out:
                fac = build_mu(fac, grid, full_levels="bottom_top_stag" in flux[D].dims)
            fac = stagger_like(fac, flux[D], cyclic=cyclic, **grid[stagger_const])

            # flux derivative
            dx = grid["D" + D]
            adv_i[D] = -diff(flux[D] * fac / mf_flx, ds, dat_mean[d], cyclic=cyc) * mapfac["X"] * mapfac["Y"] / dx

        # vertical flux
        rhod8z_m = rhod8z
        if (comp in ["mean", "trb_r"]) and hor_avg:
            rhod8z_m = avg_xy(rhod8z, avg_dims, cyclic=cyclic)
        fz = flux["Z"] * rhod8z_m
        if VAR == "W":
            adv_i["Z"] = -diff(fz, "bottom_top", grid["ZNW"]) / grid["DN"]
            # set sfc and top point correctly
            adv_i["Z"][{"bottom_top_stag": 0}] = 0.
            adv_i["Z"][{"bottom_top_stag": -1}] = (2 * fz.isel(bottom_top=-1) / grid["DN"].isel(bottom_top_stag=-2)).values

        else:
            adv_i["Z"] = -diff(fz, "bottom_top_stag", grid["ZNU"]) / grid["DNW"]

        if hor_avg:
            adv_i = avg_xy(adv_i, avg_dims, cyclic=cyclic)
            fluxes[comp] = avg_xy(fluxes[comp], avg_dims, cyclic=cyclic)

        # multiply with g so that we can then divide all tendencies by mu
        adv_i["Z"] = adv_i["Z"] * (-g)
        for d in adv_i.data_vars:
            if dz_out and (d != "Z"):
                adv_i[d] = adv_i[d] / grid["RHOD_STAG_MEAN"]
            else:
                adv_i[d] = adv_i[d] / grid["MU_STAG_MEAN"]

        adv[comp] = adv_i

    keys = adv.keys()
    adv = xr.concat(adv.values(), "comp")
    adv = adv.to_array("dir")
    adv["comp"] = list(keys)
    flux = xr.concat(fluxes.values(), "comp")
    flux["comp"] = list(fluxes.keys())

    tend_mass = None
    if calc_mass:
        # continuity equation
        dt = attrs["AVG_INTERVAL"]
        if dz_out:
            rhom = grid["RHOD_STAG_MEAN"]
            rho_tend = dat_inst["RHOD_STAG"].diff("Time") / dt
        else:
            rhom = grid["MU_STAG_MEAN"]
            rho_tend = dat_inst["MU_STAG"].diff("Time") / dt
        # only keep end points of averaging intervals
        rho_tend = rho_tend.isel(Time=slice(None, None, 2))
        if hor_avg:
            rho_tend = avg_xy(rho_tend, avg_dims, cyclic=cyclic)
        rho_tend = rho_tend / rhom
        tend_mass = adv.sel(comp="mass", drop=True)
        tend_mass = tend_mass.reindex(dir=[*adv.dir.values, "T"])
        tend_mass.loc[{"dir": "T"}] = rho_tend.transpose(*tend_mass[0].dims)
        tend_mass = tend_mass * grid["RHOD_STAG_MEAN"]
        # calculate vertical term as residual
        adv.loc["Z", "mass"] = - adv.loc["X", "mass"] - adv.loc["Y", "mass"] + rho_tend

    if not trb_exp:
        flux = flux.reindex(comp=[*flux.comp.values, "trb_r"])
        adv = adv.reindex(comp=[*adv.comp.values, "trb_r"])

    for d in flux.data_vars:
        # calculate resolved turbulent fluxes and tendencies as residual
        if (hor_avg and (d.lower() in avg_dims)) or (not trb_exp):
            flux[d].loc["trb_r"] = flux[d].loc["adv_r"] - flux[d].loc["mean"]
            adv.loc[d, "trb_r"] = adv.loc[d, "adv_r"] - adv.loc[d, "mean"]

        if VAR == "T":
            # add advection of constant base state
            for comp in ["adv_r", "mean"]:
                flux[d].loc[comp] = flux[d].loc[comp] + 300 * flux[d].loc["mass"]
                adv.loc[d, comp] = adv.loc[d, comp] + 300 * adv.loc[d, "mass"]

    flux = flux.reindex(comp=["adv_r", "mean", "trb_r"])
    adv = adv.reindex(comp=["adv_r", "mean", "trb_r"])

    return flux, adv, vmean, var_stag, corr_flx, tend_mass


def cartesian_corrections(VAR, dim_stag, corr_flx, var_stag, vmean, rhodm, grid, adv, tend, tend_mass,
                          cyclic=None, dz_out=False, hor_avg=False, avg_dims=None):
    """
    Compute cartesian corrections and apply them to advective and total tendencies.

    Parameters
    ----------
    VAR : str
        Variable to process.
    dim_stag : str
        Staggering dimension of variable.
    corr_flx : xarray Dataarray
        Cartesian correction fluxes.
    var_stag : xarray Dataset
        Averaged variable VAR staggered to the grid of the three spatial fluxes.
    vmean : xarray Dataset
        Mean velocities.
    rhodm : xarray Dataarray
        Averaged density.
    grid : xarray Dataset
        Variables related to the model grid.
    adv : xarray Dataarray
        Decomposed advective tendencies from all three directions.
    tend : xarray Dataarray
        Total tendency of VAR.
    tend_mass : xarray DataArray
        Components of continuity equation.
    cyclic : dict of booleans for xy or None
        Defines which dimensions have periodic boundary conditions.
        Use periodic boundary conditions to fill lateral boundary points.
        The default is None.
    dz_out : bool, optional
        In Cartesian correction terms: take derivatives of z out of vertical derivative.
        The default is False.
    hor_avg : bool, optional
        Average horizontally. The default is False.
    avg_dims : str or list of str, optional
        Averaging dimensions if hor_avg=True. The default is None.

    Returns
    -------
    adv : xarray Dataarray
        Decomposed advective tendencies from all three directions
        with Cartesian corrections included.
    tend : xarray Dataarray
        Total tendency of VAR with Cartesian correction included.
    corr_flx : xarray Dataarray
        Cartesian correction fluxes with mean and resolved turbulent components included.
    corr : xarray Dataarray
        Cartesian corrections as vertical derivative of correction fluxes.
    tend_mass : xarray DataArray
        Components of continuity equation with Cartesian correction included.
    """
    print("Compute Cartesian corrections")
    # decompose cartesian corrections
    kw = dict(ref=var_stag["Z"], cyclic=cyclic, **grid[stagger_const])
    rho_stag = stagger_like(rhodm, **kw)

    # total
    if hor_avg:
        corr_flx = avg_xy(corr_flx, avg_dims, cyclic=cyclic)
        rho_stag = avg_xy(rho_stag, avg_dims, cyclic=cyclic)
    corr_flx = corr_flx.expand_dims(comp=["adv_r"]).reindex(comp=["adv_r", "mean", "trb_r"])

    if tend_mass is not None:
        corr_flx = corr_flx.reindex(comp=[*corr_flx.comp.values, "mass"])

    # mean component
    for d, v in zip(xy, ["U", "V"]):
        D = d.upper()
        if dz_out:
            # alternative corrections: dz multiplied later (after taking derivative)
            corr_d = stagger_like(vmean[D], **kw)
        else:
            corr_d = -stagger_like(grid["dzdt_{}".format(d)], **kw)
        corr_flx.loc["mean", D] = rho_stag * var_stag["Z"] * corr_d
        if tend_mass is not None:
            corr_flx.loc["mass", D] = rho_stag * corr_d
    if dz_out:
        corr_flx.loc["mean", "T"] = corr_flx.loc["adv_r", "T"]
        if tend_mass is not None:
            corr_flx.loc["mass", "T"] = rho_stag
    else:
        dzdt = stagger_like(grid["dzdd"].sel(dir="T"), **kw)
        corr_flx.loc["mean", "T"] = rho_stag * dzdt * var_stag["Z"]
        if tend_mass is not None:
            corr_flx.loc["mass", "T"] = rho_stag * dzdt

    # resolved turbulence component as residual
    corr_flx.loc["trb_r"] = corr_flx.loc["adv_r"] - corr_flx.loc["mean"]

    # correction flux to tendency: take vertical derivative
    if "W" in VAR:
        corr = diff(corr_flx, "bottom_top", grid["ZNW"]) / grid["DN"]
        corr[{"bottom_top_stag": 0}] = 0.
        corr[{"bottom_top_stag": -1}] = -(2 * corr_flx.isel(bottom_top=-1) / grid["DN"].isel(bottom_top_stag=-2)).values
    else:
        corr = diff(corr_flx, "bottom_top_stag", grid["ZNU"]) / grid["DNW"]
    corr = corr * (-g) / grid["MU_STAG_MEAN"]

    if dz_out:
        corr = corr * stagger_like(grid["dzdd"], corr,
                                           cyclic=cyclic, **grid[stagger_const])
    if tend_mass is not None:
        mass_corr = corr.sel(comp="mass")
        if VAR == "T":
            for comp in ["adv_r", "mean"]:
                # add correction for constant base state
                corr.loc[comp] = corr.loc[comp] + 300 * mass_corr
                # finish residual calculation started in adv_tend
                adv.loc["Z", comp] = adv.loc["Z", comp] - 300 * mass_corr.sum("dir")
        corr = corr.reindex(comp=adv.comp.values)
        # corrections to continuity equation
        mass_corr.loc["T"] = - mass_corr.loc["T"]
        for D in corr.dir:
            tend_mass.loc[D] = tend_mass.loc[D] + mass_corr.loc[D] * grid["RHOD_STAG_MEAN"]

    # apply corrections to horizontal advection and total tendency
    for i, d in enumerate(XY):
        adv.loc[d] = adv.loc[d] + corr.sel(dir=d)
    corr.loc[:, "T"] = - corr.loc[:, "T"]
    tend = tend + corr.sel(comp="adv_r", dir="T", drop=True)

    return adv, tend, corr_flx, corr, tend_mass


def total_tendency(dat_inst, var, grid, attrs, dz_out=False,
                   hor_avg=False, avg_dims=None, cyclic=None):
    """Compute total tendency.

    Parameters
    ----------
    dat_inst : xarray Dataset
        WRF instantaneous output.
    VAR : str
        Variable to process.
    grid : xarray Dataset
        Variables related to the model grid.
    attrs : dict
        Global attributes of WRF output.
    dz_out : bool, optional
        In Cartesian correction terms: take derivatives of z out of vertical derivative.
        The default is False.
    hor_avg : bool, optional
        Average horizontally. The default is False.
    avg_dims : str or list of str, optional
        Averaging dimensions if hor_avg=True. The default is None.
    cyclic : dict of booleans for xy, optional
        Defines which dimensions have periodic boundary conditions.
        Use periodic boundary conditions to fill lateral boundary points.
        The default is None

    Returns
    -------
    tend : xarray Dataarray
        Total tendency of VAR.

    """
    # get instantaneous variable
    if var == "t":
        if attrs["USE_THETA_M"] and (not attrs["OUTPUT_DRY_THETA_FLUXES"]):
            # get or calculate moist theta
            if "THM" in dat_inst:
                vard = dat_inst["THM"]
            else:
                vard = (dat_inst["T"] + 300) * (1 + rvovrd * dat_inst["QVAPOR"]) - 300
        else:
            vard = dat_inst["T"]
        vard = vard + 300
    elif var == "q":
        vard = dat_inst["QVAPOR"]
    else:
        # momentum
        vard = dat_inst[var.upper()]

    # couple variable to rho/mu
    if dz_out:
        rho = dat_inst["RHOD_STAG"]
        rho_m = grid["RHOD_STAG_MEAN"]
    else:
        rho = dat_inst["MU_STAG"]
        rho_m = grid["MU_STAG_MEAN"]
    rvar = vard * rho

    # total tendency
    dt = attrs["AVG_INTERVAL"]
    total_tend = rvar.diff("Time") / dt
    # only keep end points of averaging intervals
    total_tend = total_tend.isel(Time=slice(None, None, 2))

    if hor_avg:
        total_tend = avg_xy(total_tend, avg_dims, cyclic=cyclic)

    total_tend = total_tend / rho_m

    return total_tend, vard

#%% main

def calc_tendencies(variables, outpath_wrf, outpath=None, budget_methods="cartesian",
                    t_avg=False, t_avg_interval=None, hor_avg=False, avg_dims=None, hor_avg_end=False,
                    skip_exist=True, chunks=None, save_output=True, return_model_output=False,
                    **load_kw):
    """Load WRF output and start tendency calculations according to the given budget methods.
       MPI processing is available via the chunks argument.

    Parameters
    ----------
    variables : list of str
        List of variables to process.
    outpath_wrf : str or path-like
        Path to the WRF output directory.
    outpath : str or path-like, optional
        Where to save the postprocessed output. Defaults to $outpath_wrf/postprocessed.
    budget_methods : str or list of str, optional
        Budget calculation methods to apply. One method is a string that contains
        keys from tools.budget_settings separated by a space.
        Several methods can be combined in a list.
        The default is "cartesian".
    t_avg : bool, optional
        Average WRF output again over time. The default is False.
    t_avg_interval : integer, optional
        Interval for time averaging (number of output time steps) if t_avg=True.
        The default is None.
    hor_avg : bool, optional
        Average horizontally. The default is False.
    avg_dims : str or list of str, optional
        Averaging dimensions if hor_avg=True. The default is None.
    hor_avg_end : bool, optional
        Do horizontal average at end of processing step instead of inside.
        This affects the mean and turbulent components. The default is False.
    skip_exist : bool, optional
        Skip variables for which postprocessed output is already available. The default is True.
    chunks : dict of integers, optional
        Mapping from dimension "x" and/or "y" to chunk sizes. The domain is split into tiles
        with the given chunk sizes which are postprocessed independently. If script is run with
        mpirun, the tiles can be processed in parallel.
        The default is None which disables chunking.
    save_output : bool, optional
        Save postprocessed output to disk. The default is True.
    return_model_output : bool, optional
        Also return WRF model output (dat_inst and dat_mean) for further calculations.
        The default is False.
    **load_kw :
        Keyword arguments passed to load_data.

    Returns
    -------
    datout : dict
        Postprocessed output as nested dictionary.
    dat_inst : xarray Dataset
        Modified WRF instantaneous output. Only for return_model_output=True.
    dat_mean : xarray Dataset
        Modified WRF time-averaged output. Only for return_model_output=True.


    """
    if hor_avg or hor_avg_end:
        avg = "_avg_" + "".join(avg_dims)
        if hor_avg_end:
            avg += "_end"
    else:
        avg = ""

    outpath_wrf = Path(outpath_wrf)
    if outpath is None:
        outpath = outpath_wrf / "postprocessed"
    else:
        outpath = Path(outpath)

    # check if postprocessed output already exists
    if skip_exist:
        skip = True
    else:
        skip = False
    cartesian = "cartesian" in " ".join(make_list(budget_methods))
    adv_form = "adv_form" in " ".join(make_list(budget_methods))

    for outfile in outfiles:
        if (outfile == "corr") and (not cartesian):
            continue
        for var in variables:
            if (outfile == "tend_mass") and (var != "t"):
                continue
            fpath = outpath / var.upper() / (outfile + avg + ".nc")
            if fpath.exists():
                if (not skip_exist) and (rank == 0):
                    os.remove(fpath)
            else:
                skip = False

    kwargs = dict(budget_methods=budget_methods, t_avg=t_avg, t_avg_interval=t_avg_interval,
                  hor_avg=hor_avg, hor_avg_end=hor_avg_end, avg_dims=avg_dims, skip_exist=skip_exist,
                  save_output=save_output, return_model_output=return_model_output, **load_kw)

    if skip:
        print("Postprocessed output already available!")
        out = {var: load_postproc(outpath, var, cartesian, adv_form, hor_avg=hor_avg,
                                  avg_dims=avg_dims, hor_avg_end=hor_avg_end) for var in variables}
        if return_model_output:
            print("Load model output")
            dat_mean, dat_inst = load_data(outpath_wrf, **load_kw)
            out = [out, dat_inst, dat_mean]
        return out

    if chunks is not None:
        # prepare tile processing

        if any([c not in xy for c in chunks.keys()]):
            raise ValueError("Chunking is only allowed in the x and y-directions! "
                             "Given chunks: {}".format(chunks))
        if hor_avg:
            if any([d in avg_dims for d in chunks.keys()]):
                raise ValueError("Averaging dimensions cannot be used for chunking!")
        kwargs["return_model_output"] = False  # model_output will be added later
        # create tiles and divide among processors
        all_tiles = create_tiles(outpath_wrf, chunks=chunks, **load_kw)
        all_tiles = [(i, t) for i, t in enumerate(all_tiles)]
        tiles = all_tiles[rank::nproc]
        # check if this processor is already finished
        done = 0
        if len(tiles) == 0:
            done = 1
            print("No tiles to process!")

        if comm is not None:
            # exclude finished processors from MPI communicator
            local_comm = comm.Split(done)
        else:
            local_comm = None

        # process tiles of this processor
        for i, (task, tile) in enumerate(tiles):
            tile = {k: v for d in tile for k, v in d.items()}
            if tile == {}:
                tile = None
                task = None
            calc_tendencies_core(variables, outpath_wrf, outpath, tile=tile,
                                 task=task, comm=local_comm, **kwargs)
            done = int(i == len(tiles) - 1)
            if done:
                print("All tiles processed!")
            if comm is not None:
                # exclude finished processors from MPI communicator
                local_comm = local_comm.Split(done)

        if comm is not None:
            # wait until all processors are finished
            comm.Barrier()

        if rank == 0:
            print("Load entire postprocessed output")
            out = {var: load_postproc(outpath, var, cartesian, adv_form, hor_avg=hor_avg,
                                      avg_dims=avg_dims, hor_avg_end=hor_avg_end) for var in variables}
            if return_model_output:
                dat_mean, dat_inst = load_data(outpath_wrf, **load_kw)
                out = [out, dat_inst, dat_mean]
            return out

    else:
        if nproc > 1:
            raise ValueError("Number of processors > 1, but chunking is disabled (chunks=None)!")
        return calc_tendencies_core(variables, outpath_wrf, outpath, **kwargs)


def calc_tendencies_core(variables, outpath_wrf, outpath, budget_methods="cartesian",
                         tile=None, task=None, comm=None, t_avg=False, t_avg_interval=None,
                         hor_avg=False, avg_dims=None, hor_avg_end=False, skip_exist=True,
                         save_output=True, return_model_output=True, **load_kw):
    """Core function of calc_tendencies. Load WRF output and start tendency calculations
       according to the given budget methods. Only process a certain tile, if desired.

    Parameters
    ----------
    variables : list of str
        List of variables to process.
    outpath_wrf : str or path-like
        Path to the WRF output directory.
    outpath : str or path-like
        Where to save the postprocessed output.
    budget_methods : str or list of str, optional
        Budget calculation methods to apply. One method is a string that contains
        keys from tools.budget_settings separated by a space.
        Several methods can be combined in a list.
        The default is "cartesian".
    tile : dict, optional
        Tile to process. Mapping from dimension names to integer-based indexers.
        The default is None.
    task : integer, optional
        ID number of current tile task. The default is None.
    comm : mpi4py.MPI.Intracomm, optional
        Intracommunicator for communication between MPI tasks. The default is None.
    t_avg : bool, optional
        Average WRF output again over time. The default is False.
    t_avg_interval : integer, optional
        Interval for time averaging (number of output time steps) if t_avg=True.
        The default is None.
    hor_avg : bool, optional
        Average horizontally. The default is False.
    avg_dims : str or list of str, optional
        Averaging dimensions if hor_avg=True. The default is None.
    hor_avg_end : bool, optional
        Do horizontal average at end of processing step instead of inside.
        This affects the mean and turbulent components. The default is False.
    skip_exist : bool, optional
        Skip variables for which postprocessed output is already available. The default is True.
    save_output : bool, optional
        Save postprocessed output to disk. The default is True.
    return_model_output : bool, optional
        Also return WRF model output (dat_inst and dat_mean) for further calculations.
        The default is False.
    **load_kw :
        Keyword arguments passed to xr.open_dataset.


    Returns
    -------
    datout : dict
        Postprocessed output as nested dictionary.
    dat_inst : xarray Dataset
        Modified WRF instantaneous output. Only for return_model_output=True.
    dat_mean : xarray Dataset
        Modified WRF time-averaged output. Only for return_model_output=True.

    """
    print("Load model output")
    outpath = Path(outpath)
    outpath_wrf = Path(outpath_wrf)
    dat_mean_all, dat_inst_all = load_data(outpath_wrf, **load_kw)
    dat_mean = dat_mean_all
    dat_inst = dat_inst_all

    # check if periodic bc can be used in staggering operations
    cyclic = {d: bool(dat_inst_all.attrs["PERIODIC_{}".format(d.upper())]) for d in xy}
    cyclic["bottom_top"] = False

    budget_methods = make_list(budget_methods)
    cartesian = "cartesian" in " ".join(budget_methods)
    adv_form = "adv_form" in " ".join(make_list(budget_methods))

    # select tile
    if tile is not None:
        print("\n\n{0}\nProcess tile: {1}\n".format("#" * 30, tile))
        dat_mean = dat_mean_all[tile]
        dat_inst = dat_inst_all[tile]
        # periodic BC cannot be used in tiling dimensions
        cyclic = {d: cyclic[d] and (d not in tile) for d in cyclic.keys()}
    if np.prod(list(dat_mean.sizes.values())) == 0:
        raise ValueError("At least one dimension is empy after indexing!")

    if hor_avg or hor_avg_end:
        avg = "_avg_" + "".join(avg_dims)
        if hor_avg_end:
            avg += "_end"
            hor_avg = False
    else:
        avg = ""

    # prepare variable tendencies
    dat_mean, dat_inst, grid, attrs = prepare(dat_mean, dat_inst, variables,
                                              cyclic=cyclic, t_avg=t_avg,
                                              t_avg_interval=t_avg_interval,
                                              hor_avg=hor_avg, avg_dims=avg_dims)
    datout_all = {}
    for var in sorted(variables):
        datout = {}
        VAR = var.upper()
        print("\n\n{0}\nProcess variable {1}\n".format("#" * 20, VAR))
        # check if postprocessed output already exists
        if skip_exist:
            skip = True
            for outfile in outfiles:
                if (outfile == "corr") and (not cartesian):
                    continue
                if (outfile == "tend_mass") and (var != "t"):
                    continue
                fpath = outpath / VAR / (outfile + avg + ".nc")
                if not fpath.exists():
                    skip = False
            if skip:
                print("Postprocessed output already available!")
                datout_all[var] = load_postproc(outpath, var, cartesian, adv_form, hor_avg=hor_avg,
                                                avg_dims=avg_dims, hor_avg_end=hor_avg_end)
                continue

        # calculate all sources except advection
        dat_mean, dat_inst, sgs, sgsflux, sources, sources_sum, grid, dim_stag, mapfac, \
            = calc_tend_sources(dat_mean, dat_inst, var, grid, cyclic, attrs,
                                hor_avg=hor_avg, avg_dims=avg_dims)

        # total and advective tendencies
        for budget_method in budget_methods:
            budget_method = budget_method.strip()
            datout_c = {}
            # get config dict for current budget method
            c, budget_method = get_budget_method(budget_method)
            print("\nBudget method: " + budget_method)
            dz_out = False
            if c["dz_out_x"] or c["dz_out_z"]:
                dz_out = True
                if not c["cartesian"]:
                    raise ValueError("dz_out can only be used for Cartesian calculations!")
                if c["dz_out_x"] and c["dz_out_z"]:
                    raise ValueError("dz_out_x and dz_out_z cannot be used at the same time!")
            total_tend, vard = total_tendency(dat_inst, var, grid, attrs, dz_out=dz_out,
                                              hor_avg=hor_avg, avg_dims=avg_dims, cyclic=cyclic)
            # advective tendency
            calc_mass = (var == "t") or (c["adv_form"] and (var == "q"))
            dat = adv_tend(dat_mean, dat_inst, VAR, grid, mapfac, cyclic, attrs,
                           hor_avg=hor_avg, avg_dims=avg_dims, calc_mass=calc_mass,
                           cartesian=c["cartesian"], force_2nd_adv=c["force_2nd_adv"],
                           dz_out_x=c["dz_out_x"], dz_out_z=c["dz_out_z"])
            if dat is None:
                continue
            else:
                datout_c["flux"], datout_c["adv"], vmean, var_stag, corr_flx, tend_mass = dat

            datout_c["net"] = total_tend

            if c["cartesian"]:
                out = cartesian_corrections(VAR, dim_stag, corr_flx, var_stag, vmean,
                                            dat_mean["RHOD_MEAN"], grid, datout_c["adv"],
                                            total_tend, tend_mass, cyclic, dz_out=dz_out,
                                            hor_avg=hor_avg, avg_dims=avg_dims)
                datout_c["adv"], datout_c["net"], corr_flx, datout_c["corr"], tend_mass = out

            if tend_mass is not None:
                datout_c["tend_mass"] = tend_mass
            # transform flux-form to advective form
            if c["adv_form"]:
                if tend_mass is None:
                    for v in ["q", "t"]:
                        try:
                            tend_mass = datout_all[v]["tend_mass"].sel(ID=budget_method)
                            break
                        except KeyError:
                            if v == "t":
                                raise ValueError("tend_mass not available! "
                                                 "You need to process variable q or t before!")
                if not c["cartesian"]:
                    raise ValueError("Advective form only implemented for Cartesian grid!")
                var_mean_c = dat_mean[VAR + "_MEAN"]
                if hor_avg:
                    var_mean_c = avg_xy(var_mean_c, avg_dims, cyclic=cyclic)
                if VAR == "T":
                    var_mean_c = var_mean_c + 300

                trans = tend_mass.copy(deep=True)
                trans.loc["Z"] = trans.loc["T"] - trans.loc["X"] - trans.loc["Y"]
                trans = stagger_like(trans, var_mean_c, cyclic=cyclic, **grid[stagger_const])
                trans = var_mean_c * trans / grid["RHOD_STAG_MEAN"]
                datout_c["net"] = datout_c["net"] - trans.sel(dir="T", drop=True) # TODO need online-averaged variable?
                datout_c["adv"].loc[{"comp": ["mean", "adv_r"]}] = datout_c["adv"].loc[{"comp": ["mean", "adv_r"]}] \
                    - trans.sel(dir=["X","Y","Z"])

            # merge resolved and sgs tendencies and fluxes
            datout_c["adv"] = datout_c["adv"].to_dataset(name="adv")
            if isinstance(sgs, xr.DataArray):
                sgs = sgs.to_dataset(name="adv")
            for sgs_dat, v in zip([sgs, sgsflux], ["adv", "flux"]):
                datout_c[v] = datout_c[v].reindex(comp=[*datout_c[v].comp.values, "trb_s"])
                for dv in datout_c[v].data_vars:
                    if c["cartesian"]:
                        IDs = "cartesian"
                    else:
                        IDs = "native"
                    datout_c[v][dv].loc[{"comp": "trb_s"}] = sgs_dat[dv].sel(ID=IDs)
            adv = datout_c["adv"].to_array().isel(variable=0, drop=True).reindex(dir=[*XYZ, "sum"])
            adv_sum = adv.sum("dir")
            adv.loc[{"dir": "sum"}] = adv_sum
            datout_c["adv"] = adv

            # add all forcings
            datout_c["forcing"] = datout_c["adv"].sel(comp="adv_r", dir="sum", drop=True) + sources_sum

            if "dim" in datout_c["net"].coords:
                # remove inappropriate coordinate
                datout_c["net"] = datout_c["net"].drop("dim")

            # aggregate output of different IDs
            loc = dict(ID=[budget_method])
            for dn in datout_c.keys():
                datout_c[dn] = datout_c[dn].expand_dims(loc)
                if dn not in datout:
                    datout[dn] = datout_c[dn]
                else:
                    datout[dn] = xr.concat([datout[dn], datout_c[dn]], "ID")

        net = datout["net"].expand_dims(side=["tendency"])
        forcing = datout["forcing"].expand_dims(side=["forcing"])
        net = xr.concat([net, forcing], "side")

        # set units and descriptions
        units = units_dict_tend[var]
        units_flx = units_dict_flux[var]
        datout["flux"] = datout["flux"].assign_attrs(description="{}-flux".format(VAR),
                                                     units=units_flx)
        datout["tend"] = sources.assign_attrs(description="{}-tendency sources".format(VAR),
                                              units=units)
        datout["tend"]["adv"] = datout["adv"].assign_attrs(description="advective {}-tendency".format(VAR),
                                                 units=units)
        datout["tend"]["net"] = net.assign_attrs(description="total {}-tendency".format(VAR),
                                                 units=units)
        del datout["forcing"], datout["net"], datout["adv"]

        if "tend_mass" in datout:
            datout["tend_mass"] = datout["tend_mass"].assign_attrs(units="kg m-3 s-1",
                   description="Components of continuity equation")
        if "corr" in datout:
            datout["corr"] = datout["corr"].assign_attrs(
                description="{}-tendency correction".format(VAR), units=units)
        grid["MU_STAG_MEAN"] = grid["MU_STAG_MEAN"].assign_attrs(
            description="time-averaged dry air mass", units="Pa")
        datout["grid"] = grid

        if save_output:
            print("\nSave data")

        for dn, dat in datout.items():
            warn_duplicate_dim(dat, name=dn)
            if hor_avg_end:
                datout[dn] = avg_xy(dat, avg_dims, cyclic=cyclic, **grid[stagger_const])

        for dn, dat in datout.items():
            # add height as coordinate
            if "flux" in dn:
                grid = datout["grid"]
                for D in XYZ:
                    z = stagger_like(grid["ZW"], dat[D], cyclic=cyclic, **grid[stagger_const])
                    z = z.assign_attrs(description=z.description + " staggered to {}-flux grid".format(D))
                    dat[D] = dat[D].assign_coords({"zf{}".format(D.lower()): z})
            elif dn != "grid":
                dat = dat.assign_coords(z=grid["Z_STAG"])
            # add global attributes
            dat = dat.assign_attrs(attrs)

            da_type = False
            if type(dat) == DataArray:
                # data needs to be a Dataset
                da_type = True
                dat = dat.to_dataset(name=dn)

            for v in dat.variables:
                # delete unnecessary variable attributes
                dat[v].attrs = {k: v for k, v in dat[v].attrs.items() if k not in del_attrs}

            if tile is not None:
                # strip tile boundary points except for domain boundary points
                t_bounds = {}
                for d, l in tile.items():
                    if d not in dat.dims:
                        continue
                    start = None
                    stop = None
                    if l.start is not None:
                        start = 1
                    if l.stop is not None:
                        if "stag" in d:
                            # for staggered dimension: cut two points at end
                            stop = -2
                        else:
                            stop = -1
                    t_bounds[d] = slice(start, stop)
                dat = dat[t_bounds]

            if save_output:
                fpath = outpath / VAR
                os.makedirs(fpath, exist_ok=True)
                fpath = fpath / (dn + avg + ".nc")
                if tile is None:
                    dat.to_netcdf(fpath)
                else:
                    save_tiles(dat, dn, fpath, dat_mean_all.coords, task, tile, comm=comm)

            if da_type:
                # convert Dataset back to DataArray
                dat = dat[dn]

            datout[dn] = dat

        if tile is None:
            datout_all[var] = datout

    if tile is None:
        out = datout_all
        if return_model_output:
            out = [out, dat_inst_all, dat_mean_all]
        return out
    # in case of tiling: load output later


# %% tile processing

def create_tiles(outpath_wrf, chunks, **load_kw):
    """Split processing domain into xy tiles according to the dictionary chunks."""
    print("Create tasks")
    dat_mean, _ = load_data(outpath_wrf, **load_kw)
    tiles = []
    # iterate over chunking dimensions
    for dim, size in chunks.copy().items():
        if dim not in dat_mean.dims:
            raise ValueError("Chunking dimension {} not in data!".format(dim))
        # split domain in tiles with given size
        bounds = np.arange(len(dat_mean[dim]))[::size]
        if len(bounds) == 1:
            print("Chunking in {0}-direction leads to one chunk only. "
                  "Deleting {0} from chunks dictionary.".format(dim))
            del chunks[dim]
            continue
        # add halo points to tiles and create indexers
        iloc = []
        for i in range(len(bounds)):
            iloc_b = {}
            for stag in [False, True]:
                if stag:
                    dim_s = dim + "_stag"
                    # staggered dimensions need one gridpoint more
                    ext = 2
                else:
                    dim_s = dim
                    ext = 1
                if i == 0:
                    # no halo points at domain start
                    start = None
                else:
                    start = bounds[i] - 1
                if i == len(bounds) - 1:
                    # no halo points at domain end:
                    stop = None
                else:
                    stop = bounds[i + 1] + ext

                iloc_b[dim_s] = slice(start, stop)
            iloc.append(iloc_b)
        tiles.append(iloc)
    # create all combinations of x slices and y slices
    tiles = list(itertools.product(*tiles))

    return tiles


def save_tiles(dat, name, fpath, coords_all, task, tile, comm=None):
    """Save current tile to disk.

    If several tiles are processed in parallel,
    the writing is also in parallel. The first task writes the
    coordinate variables and global attributes.


    Parameters
    ----------
    dat : xarray Dataset or DataArray
        Data to save.
    name : str
        Name of data to save.
    fpath : str or path-like
        Save location.
    coords_all : xarray Dataset
        Coordinates of complete file.
    tile : dict
        Tile to process. Mapping from dimension names to integer-based indexers.
    task : integer
        ID number of current tile task.
    comm : mpi4py.MPI.Intracomm, optional
        Intracommunicator for communication between MPI tasks. The default is None.

    Returns
    -------
    None.

    """
    logger = logging.getLogger('l1')
    logger.debug("#" * 20 + "\n\n Writing {}".format(name))
    if os.path.isfile(fpath):
        mode = "r+"
    else:
        mode = "w"

    if comm is None:
        parallel = False
    else:
        parallel = True

    nc = netCDF4.Dataset(fpath, mode=mode, parallel=parallel, comm=comm)
    # coordinates of whole dataset
    coords_all = {d: coords_all[d].values for d in tile.keys() if d in dat.dims}
    if mode == "w":
        tempfile = str(fpath) + ".tmp"
        if task == 0:
            # create small template file to copy attributes and coordinates from
            tmp = dat.isel({d: [1] for d in tile.keys() if d in dat.dims})
            if os.path.isfile(tempfile):
                os.remove(tempfile)
            tmp.to_netcdf(tempfile)

        nc_dat = netCDF4.Dataset(tempfile, "r", parallel=parallel, comm=comm)
        logger.debug("Loaded template nc file")

        # create dimensions
        for d in dat.dims:
            if d in coords_all:
                size = len(coords_all[d])
            else:
                size = len(dat[d])
            nc.createDimension(d, size)
        logger.debug("Created dimensions")

    general_vars = []
    for v in dat.variables:
        logger.debug("\nwrite variable {}".format(v))
        if (v in coords_all) or all([d not in tile for d in dat[v].dims]):
            # dimension coordinates and variables that are the same for all tiles
            general_vars.append(v)
        if mode == "w":
            # create variable and set attributes
            dtype = nc_dat[v].dtype
            var = nc.createVariable(v, dtype, dat[v].dims)
            attrs = {att: nc_dat[v].getncattr(att) for att in nc_dat[v].ncattrs()}
            var.setncatts(attrs)
            logger.debug("created variable and set attributes")
        if v not in general_vars:
            # fill nc file with current tile data
            loc = []
            for d in nc[v].dimensions:
                start = None
                stop = None
                if d in tile:
                    start = list(coords_all[d]).index(dat[d][0])
                    stop = start + len(dat[d])
                loc.append(slice(start, stop))
            logger.debug("set tile: {}".format(loc))
            nc[v][loc] = dat[v].values

    nc.close()
    if task == 0:
        # set general vars
        nc = netCDF4.Dataset(fpath, mode="r+")
        for v in general_vars:
            if v in coords_all:
                nc[v][:] = coords_all[v]
            else:
                nc[v][:] = np.array(nc_dat[v][:])
        # global attributes
        attrs = {att: nc_dat.getncattr(att) for att in nc_dat.ncattrs()}
        nc.setncatts(attrs)
        os.remove(tempfile)
        nc.close()

    if mode == "w":
        nc_dat.close()
