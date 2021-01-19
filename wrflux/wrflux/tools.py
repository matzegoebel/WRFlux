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
import decimal
import netCDF4
import sys
import numpy as np

import os
import pandas as pd
from datetime import datetime
from functools import partial
import itertools
try:
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

logger = logging.getLogger('l1')
logger.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# ch.setStream(sys.stdout)
# logger.addHandler(ch)
print = partial(print, flush=True)

xr.set_options(arithmetic_join="exact")
xr.set_options(keep_attrs=True)
DataArray = xr.core.dataarray.DataArray
Dataset = xr.core.dataset.Dataset

# directory for figures
if "FIGURES" in os.environ:
    figloc = os.environ["FIGURES"]
else:
    print("Environment variable FIGURES not available. Saving figures to HOME directory.")
    figloc = os.environ["HOME"]

# %% constants

dim_dict = dict(x="U", y="V", bottom_top="W", z="W")
xy = ["x", "y"]
XY = ["X", "Y"]
XYZ = [*XY, "Z"]
uvw = ["u", "v", "w"]
units_dict = {"t": "K ", "q": "", **{v: "ms$^{-1}$" for v in uvw}}
units_dict_tend = {"t": "Ks$^{-1}$", "q": "s$^{-1}$", **{v: "ms$^{-2}$" for v in uvw}}
units_dict_flux = {"t": "Kms$^{-1}$", "q": "ms$^{-1}$", **{v: "m$^{2}$s$^{-2}$" for v in uvw}}
units_dict_tend_rho = {"t": "kg m$^{-3}$Ks$^{-1}$",
                       "q": "kg m$^{-3}$s$^{-1}$", **{v: "kg m$^{-2}$s$^{-2}$" for v in uvw}}
g = 9.81
rvovrd = 461.6 / 287.04
stagger_const = ["FNP", "FNM", "CF1", "CF2", "CF3", "CFN", "CFN1"]

outfiles = ["grid", "adv", "flux", "tend", "sources", "sgs", "sgsflux", "corr"]
del_attrs = ["MemoryOrder", "FieldType", "stagger", "coordinates"]

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
    try:
        ds = xr.open_dataset(file, **kwargs)
    except ValueError as e:
        if "unable to decode time" in e.args[0]:
            ds = xr.open_dataset(file, decode_times=False, **kwargs)
        else:
            raise e
    if fix_c:
        dx, dy = ds.DX, ds.DY
        ds = fix_coords(ds, dx=dx, dy=dy)

    if del_attrs:
        ds.attrs = {}

    ds.close()

    return ds


def fix_coords(data, dx, dy):
    """Assign time and space coordinates to dataset/dataarray."""
    # assign time coordinate
    if ("XTIME" in data) and (type(data.XTIME.values[0]) == np.datetime64):
        data = data.assign_coords(Time=data.XTIME)
    else:
        time = data.Times.astype(str).values
        time = pd.DatetimeIndex([datetime.fromisoformat(str(t)) for t in time])
        data = data.assign_coords(Time=time)

    for v in ["XTIME", "Times"]:
        if v in data:
            data = data.drop(v)
    # assign x and y coordinates and rename dimensions
    for dim_old, res, dim_new in zip(["south_north", "west_east"], [dy, dx], ["y", "x"]):
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
    if ("ZNW" in data) and ("bottom_top_stag" in data.dims):
        data = data.assign_coords(bottom_top_stag=data["ZNW"].isel(Time=0, drop=True))
    if ("ZNU" in data) and ("bottom_top" in data.dims):
        data = data.assign_coords(bottom_top=data["ZNU"].isel(Time=0, drop=True))

    return data


# %%misc functions

def make_list(o):
    """Convert object to list if it is not already a tuple, list, dictionary, or array."""
    if type(o) not in [tuple, list, dict, np.ndarray]:
        o = [o]
    return o


def coarsen_avg(data, dim, interval, rho=None, cyclic=None,
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
    cyclic : dict of booleans for all dimensions or None, optional
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

    out = xr.Dataset()
    for var in data.data_vars:
        if (rho is not None) and (var in rho_weighted_vars):
            rho_s = stagger_like(rho, data[var], cyclic=cyclic, **stagger_kw)
            rho_s_mean = rho_s.coarsen(**avg_kwargs).mean()
            out[var] = (rho_s * data[var]).coarsen(**avg_kwargs).mean() / rho_s_mean
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
    cyclic : dict of booleans for all dimensions or None, optional
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
    # TODOm: bottleneck?
    if type(data) == Dataset:
        out = xr.Dataset()
        for v in data.data_vars:
            out[v] = avg_xy(data[v], avg_dims, rho=rho, cyclic=cyclic, **stagger_const)
        return out

    if rho is not None:
        rho_s = stagger_like(rho, data, cyclic=cyclic, **stagger_const)
        for d in rho_s.dims:
            # TODOm: slow?
            if (d not in rho.dims) and ("bottom_top" not in d) and (not cyclic[d[0]]):
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

        # cut boundary points depending on lateral BC
        if (cyclic is None) or (not cyclic[d]):
            data = loc_data(data, iloc={d: slice(1, -1)})
            if rho is not None:
                rho_s = loc_data(rho_s, iloc={d: slice(1, -1)})
        elif ds in data.dims:
            data = data[{ds: slice(0, -1)}]
            if rho is not None:
                rho_s = rho_s[{ds: slice(0, -1)}]

    # do (density-weighted) average
    if rho is None:
        return data.mean(avg_dims_final)
    else:
        return (rho_s * data).mean(avg_dims_final) / rho_s_mean


def find_bad(dat, nan=True, inf=True):
    """Drop all indeces of each dimension in DataArray dat that do not contain any NaNs or infs."""
    # set coordinates for all dims
    for d in dat.dims:
        if d not in dat.coords:
            dat = dat.assign_coords({d: dat[d]})

    nans = False
    infs = False
    if nan:
        nans = dat.isnull()
    if inf:
        infs = dat == np.inf
    invalid = nans | infs
    invalid = invalid.where(invalid)
    invalid = dropna_dims(invalid)
    if invalid.size > 0:
        dat = dat.loc[invalid.indexes]
    else:
        dat = None
    return dat


def dropna_dims(dat, dims=None, how="all", **kwargs):
    """
    Consecutively drop NaNs along given dimensions.

    Parameters
    ----------
    dat : xarray dataset or dataarray
        input data.
    dims : iterable, optional
        dimensions to use. The default is None, which takes all dimensions.
    how : str, optional
        drop index if "all" or "any" NaNs occur. The default is "all".
    **kwargs : keyword arguments
        kwargs for dropna.

    Returns
    -------
    dat : xarray dataset or dataarray
        reduced data.

    """
    if dims is None:
        dims = dat.dims
    for d in dims:
        dat = dat.dropna(d, how=how, **kwargs)

    return dat


def max_error_scaled(dat, ref, dim=None):
    """
    Compute maximum squared error of the input data with respect to the reference data
    and scale by the variance of the reference data. If dim is given the variance and
    maximum error is computed over these dimensions. Then the maximum of the result is taken.

    Parameters
    ----------
    dat : dataarray
        input data.
    ref : dataarray
        reference data.
    dim : str or sequence of str, optional
        Dimension(s) over which to calculate variance and maximum error.
        The default is None, which means all dimensions.

    Returns
    -------
    float
        maximum scaled error.

    """
    if dim is not None:
        dim = correct_dims_stag_list(dim, ref)

    err = (dat - ref)**2
    norm = ((ref - ref.mean(dim=dim))**2).mean(dim=dim)
    err = err.max(dim=dim) / norm
    if err.shape != ():
        err = err.max()
    return float(err)


def nse(dat, ref, dim=None):
    """
    Nash–Sutcliffe efficiency coefficient.

    Parameters
    ----------
    dat : datarray
        input data.
    ref : datarray
        reference data.
    dim : str or list, optional
        dimensions along which to calculate the index.
        The default is None, which means all dimensions.

    Returns
    -------
    datarray
        nse.

    """
    if dim is not None:
        d = dict(dim=correct_dims_stag_list(dim, ref))
    else:
        d = {}
    mse = ((dat - ref)**2).mean(**d)
    norm = ((ref - ref.mean(**d))**2).mean(**d)
    return 1 - mse / norm


def warn_duplicate_dim(data, name=None):
    """Warn if dataarray or dataset contains both,
    the staggered and unstaggered version of any dimension"""
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


def rolling_mean(ds, dim, window, periodic=True, center=True):
    """
    Rolling mean over given dimension.

    Parameters
    ----------
    ds : dataarray or dataset
        input data.
    dim : str
        dimension.
    window : int
        window size of rolling mean.
    periodic : bool, optional
       Fill values close to the boundaries using periodic boundary conditions.
       The default is True.
    center : bool, optional
       Set the labels at the center of the window. The default is True.

    Returns
    -------
    ds : dataarray or dataset
        averaged data.

    """
    if periodic:
        # create pad
        if center:
            pad = int(np.floor(window / 2))
            pad = (pad, pad)
        else:
            pad = (window - 1, 0)
        ds = ds.pad({dim: pad}, mode='wrap')

    ds = ds.rolling({dim: window}, center=center).mean()
    if periodic:
        # remove pad
        ds = ds.isel({dim: np.arange(pad[0], len(ds[dim]) - pad[1])})
    return ds


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

    if iloc is not None:
        iloc = correct_dims_stag(iloc, dat)
        dat = dat[iloc]
    if loc is not None:
        loc = correct_dims_stag(loc, dat)
        dat = dat.loc[loc]
    if copy:
        dat = dat.copy()

    return dat


def round_sig(number, figures):
    number = decimal.Decimal(number)
    return format(round(number, -number.adjusted() - 1 + figures), "f")


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
    cyclic : dict of booleans for all dimensions or None, optional
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
                if (cyclic is not None) and (d in cyclic):
                    cyc = cyclic[d]
                else:
                    cyc = False
                data = stagger(data, d, ref[d + "_stag"], rename=rename, cyclic=cyc, **stagger_kw)

    return data


def stagger(data, dim, new_coord, FNM=0.5, FNP=0.5, rename=True, cyclic=False, **interp_const):
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
                             data=data, cyclic=cyclic, **interp_const)

    return data_stag


def post_stagger(data_stag, dim, new_coord, rename=True, data=None, cyclic=False, **interp_const):
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
            data_stag[{dim_s: 0}] = interp_const["CF1"] * data[{dim: 0}] + \
                interp_const["CF2"] * data[{dim: 1}] + interp_const["CF3"] * data[{dim: 2}]
            data_stag[{dim_s: -1}] = interp_const["CFN"] * \
                data[{dim: -1}] + interp_const["CFN1"] * data[{dim: -2}]
    elif cyclic:
        # set second boundary point equal to first
        data_stag.loc[{dim_s: c[-1]}] = data_stag.loc[{dim_s: c[0]}]
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


def diff(data, dim, new_coord, rename=True, cyclic=False):
    """
    Calculate first order differences along given dimension and assign new coordinates.

    Parameters
    ----------
    data : xarray dataarray
        input data.
    dim : str
        dimension over which to calculate the finite difference.
    new_coord : array-like
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
        data_s = data.shift({dim: 1})
    else:
        data_s = data.roll({dim: 1}, roll_coords=False)

    out = data - data_s
    if "_stag" in dim:
        out = out.sel({dim: out[dim][1:]})
        new_dim = dim
        if rename and (dim != "Time"):
            new_dim = dim[:dim.index("_stag")]
            out = out.rename({dim: new_dim})
        out[new_dim] = new_coord
    else:
        out = post_stagger(out, dim, new_coord, rename=rename, cyclic=cyclic)

    return out


def remove_deprecated_dims(ds):
    """Remove dimensions that do not occur in any of the variables of the given dataset."""
    var_dims = []
    for v in ds.data_vars:
        var_dims.extend(ds[v].dims)

    for d in ds.dims:
        if d not in var_dims:
            ds = ds.drop(d)
    return ds


# %%prepare tendencies


def load_data(outpath, inst_file=None, mean_file=None, start_time=None,
              pre_loc=None, pre_iloc=None, **kw):
    if inst_file is None:
        if start_time is None:
            raise ValueError("Either inst_file or start_time must be given!")
        inst_file = "instout_d01_" + start_time
    if mean_file is None:
        if start_time is None:
            raise ValueError("Either mean_file or start_time must be given!")
        mean_file = "meanout_d01_" + start_time
    fpath = outpath + "/"

    dat_inst = open_dataset(fpath + inst_file, cache=False, del_attrs=False, **kw)
    dat_mean = open_dataset(fpath + mean_file, cache=False, **kw)

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

    dims = ["Time", "bottom_top", "bottom_top_stag", "soil_layers_stag",
            "y", "y_stag", "x", "x_stag", "seed_dim_stag"]
    dat_mean = dat_mean.transpose(*[d for d in dims if d in dat_mean.dims])
    dat_inst = dat_inst.transpose(*[d for d in dims if d in dat_inst.dims])

    return dat_mean, dat_inst


def load_postproc(outpath, var, avg=None):
    datout = {}
    if avg is None:
        avg = ""
    outpath = os.path.join(outpath, "postprocessed", var.upper())
    for f in outfiles:
        file = "{}/{}{}.nc".format(outpath, f, avg)
        if f in ["sgsflux", "flux", "grid"]:
            datout[f] = xr.open_dataset(file, cache=False)
        else:
            datout[f] = xr.open_dataarray(file, cache=False)
    return datout


def get_comb(comb):
    """Build ID and settings dictionary from list of settings. Replace abbreviations."""
    keys = ["cartesian", "correct", "dz_out",
            "force_2nd_adv", "corr_varz"]  # available settings
    short_names = {"2nd": "force_2nd_adv", "corr": "correct"}  # abbreviations for settings

    if len(comb) == 0:
        IDc = "native"
    else:
        IDc = " ".join(comb)

    for i, key in enumerate(comb):
        if key in short_names:
            comb[i] = short_names[key]

    c = {}
    undefined = [key for key in comb if key not in keys]
    if len(undefined) > 0:
        raise ValueError("Undefined keys: {}".format(", ".join(undefined)))
    for k in keys:
        if k in comb:
            c[k] = True
        else:
            c[k] = False
    return c, comb, IDc


def prepare(dat_mean, dat_inst, variables=None, cyclic=None,
            t_avg=False, t_avg_interval=None, hor_avg=False, avg_dims=None):

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
    grid["DN"] = grid["DN"].rename(bottom_top="bottom_top_stag").assign_coords(
        bottom_top_stag=grid["ZNW"][:-1]).reindex(bottom_top_stag=grid["ZNW"])
    grid["DX"] = attrs["DX"]
    grid["DY"] = attrs["DY"]

    dat_mean = dat_mean.assign_coords(bottom_top=grid["ZNU"], bottom_top_stag=grid["ZNW"])
    dat_inst = dat_inst.assign_coords(bottom_top=grid["ZNU"], bottom_top_stag=grid["ZNW"])

    dat_mean = dat_mean.rename(ZWIND_MEAN="W_MEAN")
    rhod = stagger(dat_mean["RHOD_MEAN"], "bottom_top",
                   dat_mean["bottom_top_stag"], **grid[stagger_const])
    dat_mean["OMZN_MEAN"] = dat_mean["WW_MEAN"] / (-g * rhod)

    if t_avg:
        inst = dat_mean.copy()
        print("Average dat_mean over {} output steps".format(t_avg_interval))
        dat_mean = coarsen_avg(dat_mean, dim="Time", interval=t_avg_interval,
                               rho=dat_mean["RHOD_MEAN"], cyclic=cyclic,
                               stagger_kw=grid[stagger_const])

        # compute resolved turbulent fluxes explicitly if output contains all timesteps
        dt_out = float(inst.Time[1] - inst.Time[0]) / 1e9
        if round(dt_out) == attrs["DT"]:
            print("Compute turbulent fluxes explicitly")
            trb_fluxes(dat_mean, inst, variables, grid, cyclic,
                       t_avg_interval=t_avg_interval, hor_avg=hor_avg, avg_dims=avg_dims)

    # select start and end points of averaging intervals
    dat_inst = dat_inst.sel(Time=[dat_inst.Time[0].values, *dat_mean.Time.values])
    for v in dat_inst.coords:
        if ("XLAT" in v) or ("XLONG" in v):
            dat_inst = dat_inst.drop(v)

    return dat_mean, dat_inst, grid, attrs


def build_mu(mut, grid, full_levels=False):
    if full_levels:
        mu = grid["C1F"] * mut + grid["C2F"]
    else:
        mu = grid["C1H"] * mut + grid["C2H"]
    return mu


def trb_fluxes(dat_mean, inst, variables, grid, cyclic, t_avg_interval,
               hor_avg=False, avg_dims=None):

    avg_kwargs = {"Time": t_avg_interval, "coord_func": {
        "Time": partial(select_ind, indeces=-1)}, "boundary": "trim"}

    # define all needed variables
    all_vars = ["RHOD_MEAN", "OMZN_MEAN"]
    for var in variables:
        for d, vel in zip(XYZ, uvw):
            all_vars.append(var.upper() + d + "_MEAN")
            all_vars.append(vel.upper() + "_MEAN")

    # fill all time steps with block average
    means = dat_mean[all_vars].reindex(Time=inst.Time).bfill("Time")
    if hor_avg:
        means = avg_xy(means, avg_dims, rho=means["RHOD_MEAN"],
                       cyclic=cyclic, **grid[stagger_const])
    for var in variables:
        var = var.upper()
        for d, vel in zip(["X", "Y", "Z", "Z"], ["U", "V", "W", "OMZN"]):
            var_d = var + d + "_MEAN"
            vel_m = vel + "_MEAN"
            # compute perturbations
            var_pert = inst[var_d] - means[var_d]
            vel_pert = stagger_like(inst[vel_m] - means[vel_m], var_pert,
                                    cyclic=cyclic, **grid[stagger_const])
            rho_stag = stagger_like(inst["RHOD_MEAN"], var_pert,
                                    cyclic=cyclic, **grid[stagger_const])
            rho_stag_mean = stagger_like(dat_mean["RHOD_MEAN"], var_pert,
                                         cyclic=cyclic, **grid[stagger_const])
            flux = rho_stag * vel_pert * var_pert
            flux = flux.coarsen(**avg_kwargs).mean() / rho_stag_mean
            if hor_avg:
                flux = avg_xy(flux, avg_dims, cyclic=cyclic)
            dat_mean["F{}{}_TRB_MEAN".format(var, vel)] = flux


# %% WRF tendencies


def calc_tend_sources(dat_mean, dat_inst, var, grid, cyclic, attrs, hor_avg=False, avg_dims=None):
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

    dat_mean["FUY_SGS_MEAN"] = dat_mean["FVX_SGS_MEAN"]

    # density and dry air mass
    mu = grid["C2H"] + grid["C1H"] * (dat_inst["MU"] + dat_inst["MUB"])
    dat_inst["MU_STAG"] = mu
    grid["MU_STAG_MEAN"] = grid["C2H"] + grid["C1H"] * dat_mean["MUT_MEAN"]
    rhodm = dat_mean["RHOD_MEAN"]
    if var in uvw:
        rhodm = stagger(rhodm, dim_stag, dat_inst[dim_stag + "_stag"],
                        cyclic=cyclic[dim_stag], **grid[stagger_const])
        if var == "w":
            dat_inst["MU_STAG"] = grid["C2F"] + grid["C1F"] * (dat_inst["MU"] + dat_inst["MUB"])
            grid["MU_STAG_MEAN"] = grid["C2F"] + grid["C1F"] * dat_mean["MUT_MEAN"]
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
        du = d.upper()
        dzdd[du] = diff(dat_mean["Z_MEAN"], d, dat_mean[d + "_stag"],
                        cyclic=cyclic[d]) / grid["D" + du]

    zw_inst = (dat_inst["PH"] + dat_inst["PHB"]) / g
    dt = int(dat_inst.Time[1] - dat_inst.Time[0]) * 1e-9
    dzdd["T"] = zw_inst.diff("Time") / dt
    for d in [*XY, "T"]:
        dzdd[d] = stagger_like(dzdd[d], ref, ignore=["bottom_top_stag"], cyclic=cyclic)
    dzdd = remove_deprecated_dims(dzdd)
    grid["dzdd"] = dzdd.to_array("dir").assign_attrs(
        description="derivative of geopotential height with respect to x, y, and t")
    for d, vel in zip(XY, ["u", "v"]):
        dz = dat_mean["DPH_{}_MEAN".format(d)] / g
        dzdt_d = stagger_like(dz, ref, ignore=["bottom_top_stag"], cyclic=cyclic)
        desc = dzdt_d.description.replace("tendency", "height tendency")
        grid["dzdt_{}".format(d.lower())] = dzdt_d.assign_attrs(description=desc,
                                                                units="m s-1")
    rhod = - 1 / diff(g * zw_inst, "bottom_top_stag", dat_inst.bottom_top) * grid["DNW"] * mu
    dat_inst["RHOD_STAG"] = stagger_like(rhod, ref, cyclic=cyclic, **grid[stagger_const])

    # height
    grid["ZW"] = dat_mean["Z_MEAN"]
    grid["Z_STAG"] = stagger_like(dat_mean["Z_MEAN"], ref, cyclic=cyclic, **grid[stagger_const])

    # additional sources
    print("Compute SGS and additional tendencies")

    sources = xr.Dataset()
    if var == "t":
        sources["mp"] = dat_mean["T_TEND_MP_MEAN"]
        sources["rad_lw"] = dat_mean["T_TEND_RADLW_MEAN"]
        sources["rad_sw"] = dat_mean["T_TEND_RADSW_MEAN"]
        if attrs["USE_THETA_M"] and (not attrs["OUTPUT_DRY_THETA_FLUXES"]):
            # convert sources from dry to moist theta
            sources = sources * (1 + rvovrd * dat_mean["Q_MEAN"])
            # add mp tendency
            sources["mp"] = sources["mp"] + dat_mean["Q_TEND_MP_MEAN"] * rvovrd * dat_mean["T_MEAN"]
    elif var == "q":
        sources["mp"] = dat_mean["Q_TEND_MP_MEAN"]
    else:
        sources["pg"] = dat_mean["{}_TEND_PG_MEAN".format(VAR)]
        sources["cor_curv"] = dat_mean["{}_TEND_COR_CURV_MEAN".format(VAR)]

    # calculate tendencies from sgs fluxes and corrections
    sgs, sgsflux = sgs_tendency(dat_mean, VAR, grid, cyclic, dim_stag=dim_stag, mapfac=mapfac)

    if hor_avg:
        sources = avg_xy(sources, avg_dims, cyclic=cyclic)
        sgs = avg_xy(sgs, avg_dims, cyclic=cyclic)
        sgsflux = avg_xy(sgsflux, avg_dims, cyclic=cyclic)
        grid = avg_xy(grid, avg_dims, cyclic=cyclic)

    sources = sources.to_array("comp")
    sources_sum = sources.sum("comp") + sgs.sum("dir", skipna=False)

    return dat_mean, dat_inst, sgs, sgsflux, sources, sources_sum, grid, dim_stag, mapfac


def sgs_tendency(dat_mean, VAR, grid, cyclic, dim_stag=None, mapfac=None):
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
    rhoz = stagger_like(dat_mean["RHOD_MEAN"], fz, cyclic=cyclic, **grid[stagger_const])
    sgs["Z"] = -diff(fz * rhoz, d3s, new_coord=vcoord)
    sgs["Z"] = sgs["Z"] / dn / grid["MU_STAG_MEAN"] * (-g)
    for d, v in zip(xy, ["U", "V"]):
        # compute corrections
        du = d.upper()
        if mapfac is None:
            m = 1
        else:
            m = mapfac[du]
        fd = dat_mean["F{}{}_SGS_MEAN".format(VAR, du)]
        sgsflux[du] = fd
        fd = fd * stagger_like(dat_mean["RHOD_MEAN"], fd, cyclic=cyclic, **grid[stagger_const])
        cyc = cyclic[d]
        if d in fd.dims:
            # for momentum variances
            ds = d
            d = ds + "_stag"
            flux8v = stagger(fd, ds, new_coord=sgs[d], cyclic=cyc)
        else:
            ds = d + "_stag"
            flux8v = destagger(fd, ds, new_coord=sgs[d])

        if VAR == "W":
            flux8z = destagger(flux8v, d3, grid["ZNU"])
        else:
            flux8z = stagger(flux8v, d3, grid["ZNW"], **grid[stagger_const])
            flux8z[:, [0, -1]] = 0
        corr_sgs = diff(flux8z, d3s, new_coord=vcoord) / dn
        corr_sgs = corr_sgs * stagger_like(grid["dzdd"].loc[du], corr_sgs, cyclic=cyclic, **grid[stagger_const])

        dx = grid["D" + du]
        sgs[du] = -diff(fd, ds, new_coord=sgs[d], cyclic=cyc) / dx * m
        if VAR == "W":
            m = mapfac["Y"]
        sgs[du] = sgs[du] / grid["RHOD_STAG_MEAN"] + corr_sgs * m / grid["MU_STAG_MEAN"] * (-g)

    sgsflux["Z"] = fz
    sgs = sgs[XYZ]
    sgs = sgs.to_array("dir")
    if VAR == "W":
        sgs[:, :, [0, -1]] = 0

    return sgs, sgsflux


def adv_tend(dat_mean, VAR, grid, mapfac, cyclic, attrs, hor_avg=False, avg_dims=None,
             cartesian=False, force_2nd_adv=False, dz_out=False, corr_varz=False):

    print("Compute resolved tendencies")

    # get appropriate staggered variables, vertical velocity, and flux variables
    var_stag = xr.Dataset()
    fluxnames = ["F{}{}_ADV_MEAN".format(VAR, d) for d in XYZ]
    if force_2nd_adv:
        fluxnames = [fn + "_2ND" for fn in fluxnames]
        for d, f in zip(XYZ, fluxnames):
            var_stag[d] = stagger_like(dat_mean["{}_MEAN".format(VAR)],
                                       dat_mean[f], cyclic=cyclic, **grid[stagger_const])
        if VAR == "T":
            var_stag = var_stag - 300
            if attrs["USE_THETA_M"] and (not attrs["OUTPUT_DRY_THETA_FLUXES"]):
                raise ValueError(
                    "Averaged moist potential temperature not available to build "
                    "mean 2nd-order fluxes! (use_theta_m=1 and output_dry_theta_fluxes=0)")
    else:
        for d in XYZ:
            var_stag[d] = dat_mean["{}{}_MEAN".format(VAR, d)]

    if cartesian:
        w = dat_mean["WD_MEAN"]
    else:
        w = dat_mean["OMZN_MEAN"]

    if not all([f in dat_mean for f in fluxnames]):
        raise ValueError("Fluxes not available!")

    vmean = xr.Dataset({"X": dat_mean["U_MEAN"], "Y": dat_mean["V_MEAN"], "Z": w})
    if hor_avg:
        var_stag = avg_xy(var_stag, avg_dims,
                          rho=dat_mean["RHOD_MEAN"], cyclic=cyclic, **grid[stagger_const])
        for k in vmean.keys():
            vmean[k] = avg_xy(vmean[k], avg_dims, rho=dat_mean["RHOD_MEAN"],
                              cyclic=cyclic, **grid[stagger_const])

    tot_flux = dat_mean[fluxnames]
    tot_flux = tot_flux.rename(dict(zip(fluxnames, XYZ)))
    rhod8z = stagger_like(dat_mean["RHOD_MEAN"], tot_flux["Z"],
                          cyclic=cyclic, **grid[stagger_const])

    corr = ["F{}X_CORR".format(VAR), "F{}Y_CORR".format(VAR), "CORR_D{}DT".format(VAR)]
    if force_2nd_adv:
        corr = [corri + "_2ND" for corri in corr]
    corr = dat_mean[corr]
    corr = corr.to_array("dir")
    corr["dir"] = ["X", "Y", "T"]

    if not cartesian:
        tot_flux["Z"] = tot_flux["Z"] - (corr.loc["X"] + corr.loc["Y"] + corr.loc["T"]) / rhod8z
        tot_flux = tot_flux.drop("dir")
    if dz_out:
        if corr_varz:
            corr.loc["X"] = dat_mean["F{}X_CORR_DZOUT".format(VAR)]
            corr.loc["Y"] = dat_mean["F{}Y_CORR_DZOUT".format(VAR)]
        else:
            corr = tot_flux[XY]
            corr = rhod8z * stagger_like(corr, rhod8z, cyclic=cyclic, **grid[stagger_const])
            corr["T"] = corr["X"]
            corr = corr.to_array("dir")
        corr_t = dat_mean[VAR + "_MEAN"]
        corr_t = rhod8z * stagger_like(corr_t, rhod8z, cyclic=cyclic, **grid[stagger_const])
        corr.loc["T"] = corr_t

    #  mean advective fluxes
    mean_flux = xr.Dataset()
    for d in XYZ:
        if hor_avg and (d.lower() in avg_dims):
            mean_flux[d] = 0.
        else:
            vel_stag = stagger_like(vmean[d], ref=var_stag[d], cyclic=cyclic, **grid[stagger_const])
            if (VAR == "W") and (d in XY):
                vel_stag[{"bottom_top_stag": 0}] = 0
            mean_flux[d] = var_stag[d] * vel_stag

    # advective tendency from fluxes
    adv = {}
    fluxes = {"adv_r": tot_flux, "mean": mean_flux}
    try:
        # explicit resolved turbulent fluxes
        trb_flux = xr.Dataset()
        if cartesian:
            vels = ["U", "V", "W"]
        else:
            vels = ["U", "V", "OMZN"]
        for d, vel in zip(XYZ, vels):
            trb_flux[d] = dat_mean["F{}{}_TRB_MEAN".format(VAR, vel)]
        fluxes["trb_r"] = trb_flux
        trb_exp = True
    except KeyError:
        trb_exp = False
        pass

    for comp, flux in fluxes.items():
        adv_i = xr.Dataset()
        mf = mapfac
        rhod8z_m = rhod8z
        if (comp in ["trb_r", "mean"]) and hor_avg:
            mf = avg_xy(mapfac, avg_dims, cyclic=cyclic)
            rhod8z_m = avg_xy(rhod8z, avg_dims, cyclic=cyclic)
        for d in xy:
            du = d.upper()
            cyc = cyclic[d]
            if hor_avg and (d in avg_dims) and (comp in ["trb_r", "mean"]):
                adv_i[du] = 0.
                continue
            if d in flux[du].dims:
                ds = d
                d = d + "_stag"
            else:
                ds = d + "_stag"
            dx = grid["D" + du]

            mf_flx = mapfac["F" + du]

            if dz_out:
                fac = dat_mean["RHOD_MEAN"]
            else:
                fac = dat_mean["MUT_MEAN"]
            if (comp in ["trb_r", "mean"]) and hor_avg:  # TODOm: correct?
                mf_flx = avg_xy(mf_flx, avg_dims, cyclic=cyclic)
                fac = avg_xy(fac, avg_dims, cyclic=cyclic)
            if not dz_out:
                fac = build_mu(fac, grid, full_levels="bottom_top_stag" in flux[du].dims)
            fac = stagger_like(fac, flux[du], cyclic=cyclic, **grid[stagger_const])
            adv_i[du] = -diff(fac * flux[du] / mf_flx, ds, dat_mean[d], cyclic=cyc) * mf["X"] * mf["Y"] / dx
        fz = rhod8z_m * flux["Z"]
        if VAR == "W":
            adv_i["Z"] = -diff(fz, "bottom_top", grid["ZNW"]) / grid["DN"]
            # set sfc and top point correctly
            adv_i["Z"][{"bottom_top_stag": 0}] = 0.
            adv_i["Z"][{"bottom_top_stag": -1}] = (2 * fz.isel(bottom_top=-1) / grid["DN"][-2]).values

        else:
            adv_i["Z"] = -diff(fz, "bottom_top_stag", grid["ZNU"]) / grid["DNW"]
        adv_i["Z"] = adv_i["Z"] * (-g)
        for d in adv_i.data_vars:
            if dz_out and (d != "Z"):
                adv_i[d] = adv_i[d] / grid["RHOD_STAG_MEAN"]
            else:
                adv_i[d] = adv_i[d] / grid["MU_STAG_MEAN"]

        adv[comp] = adv_i

    if hor_avg:
        adv["adv_r"] = avg_xy(adv["adv_r"], avg_dims, cyclic=cyclic)
        fluxes["adv_r"] = avg_xy(fluxes["adv_r"], avg_dims, cyclic=cyclic)

    keys = adv.keys()
    adv = xr.concat(adv.values(), "comp")
    adv = adv.to_array("dir")
    adv["comp"] = list(keys)
    flux = xr.concat(fluxes.values(), "comp")
    flux["comp"] = list(fluxes.keys())

    # resolved turbulent fluxes and tendencies
    if not trb_exp:
        flux = flux.reindex(comp=["adv_r", "mean", "trb_r"])
        adv = adv.reindex(comp=["adv_r", "mean", "trb_r"])
        for d in flux.data_vars:
            flux[d].loc["trb_r"] = flux[d].loc["adv_r"] - flux[d].loc["mean"]
            adv.loc[d, "trb_r"] = adv.loc[d, "adv_r"] - adv.loc[d, "mean"]
    elif hor_avg:
        # assign fluxes in averaged dimension to turbulent component
        for d in avg_dims:
            d = d.upper()
            flux[d].loc["trb_r"] = flux[d].loc["adv_r"] - flux[d].loc["mean"]
            adv.loc[d, "trb_r"] = adv.loc[d, "adv_r"] - adv.loc[d, "mean"]

    return flux, adv, vmean, var_stag, corr


def cartesian_corrections(VAR, dim_stag, corr, var_stag, vmean, rhodm, grid, adv, tend,
                          cyclic, dz_out=False, hor_avg=False, avg_dims=None):
    print("Compute Cartesian corrections")
    # decompose cartesian corrections
    # total
    corr = corr.expand_dims(comp=["adv_r"]).reindex(comp=["adv_r", "mean", "trb_r"])
    if hor_avg:
        corr = avg_xy(corr, avg_dims, cyclic=cyclic)
        rhodm = avg_xy(rhodm, avg_dims, cyclic=cyclic)

    # mean part
    kw = dict(ref=var_stag["Z"], cyclic=cyclic, **grid[stagger_const])
    rho_stag = stagger_like(rhodm, **kw)
    for d, v in zip(xy, ["U", "V"]):
        # staggering
        du = d.upper()
        if hor_avg and (d in avg_dims):
            corr.loc["mean", du] = 0
            continue

        if dz_out:
            corr_d = stagger_like(vmean[du], **kw)
        else:
            corr_d = -stagger_like(grid["dzdt_{}".format(d)], **kw)
        corr.loc["mean", du] = corr_d * rho_stag * var_stag["Z"]

    dzdt = stagger_like(grid["dzdd"].loc["T"], **kw)
    corr.loc["mean", "T"] = rho_stag * dzdt * var_stag["Z"]

    # resolved turbulent part
    corr.loc["trb_r"] = corr.loc["adv_r"] - corr.loc["mean"]

    # correction flux to tendency
    if "W" in VAR:
        dcorr_dz = diff(corr, "bottom_top", grid["ZNW"]) / grid["DN"]
        dcorr_dz[{"bottom_top_stag": 0}] = 0.
        dcorr_dz[{"bottom_top_stag": -1}] = -(2 * corr.isel(bottom_top=-1) / grid["DN"][-2]).values
    else:
        dcorr_dz = diff(corr, "bottom_top_stag", grid["ZNU"]) / grid["DNW"]
    dcorr_dz = dcorr_dz * (-g) / grid["MU_STAG_MEAN"]

    if dz_out:
        dcorr_dz = dcorr_dz * stagger_like(grid["dzdd"], dcorr_dz,
                                           cyclic=cyclic, **grid[stagger_const])

    # apply corrections
    for i, d in enumerate(XY):
        adv.loc[d] = adv.loc[d] + dcorr_dz[:, i]

    tend = tend - dcorr_dz.sel(comp="adv_r", dir="T", drop=True)

    return adv, tend, dcorr_dz


def total_tendency(dat_inst, var, grid, dz_out=False,
                   hor_avg=False, avg_dims=None, cyclic=None, **attrs):
    # instantaneous variable
    if var == "t":
        if attrs["USE_THETA_M"] and (not attrs["OUTPUT_DRY_THETA_FLUXES"]):
            if "THM" in dat_inst:
                vard = dat_inst["THM"]
            else:
                vard = (dat_inst["T"] + 300) * (1 + rvovrd * dat_inst["QVAPOR"]) - 300
        else:
            vard = dat_inst["T"]
    elif var == "q":
        vard = dat_inst["QVAPOR"]
    else:
        vard = dat_inst[var.upper()]

    # couple variable to mu
    if dz_out:
        rvar = vard * dat_inst["RHOD_STAG"]
    else:
        rvar = vard * dat_inst["MU_STAG"]

    # total tendency
    dt = int(dat_inst.Time[1] - dat_inst.Time[0]) * 1e-9
    total_tend = rvar.diff("Time") / dt

    if hor_avg:
        total_tend = avg_xy(total_tend, avg_dims, cyclic=cyclic)

    if dz_out:
        total_tend = total_tend / grid["RHOD_STAG_MEAN"]
    else:
        total_tend = total_tend / grid["MU_STAG_MEAN"]

    return total_tend


def calc_tendencies(variables, outpath, inst_file=None, mean_file=None, start_time=None,
                    budget_methods="castesian correct", pre_iloc=None, pre_loc=None,
                    t_avg=False, t_avg_interval=None, hor_avg=False, avg_dims=None,
                    skip_exist=True, chunks=None, save_output=True, return_model_output=False,
                    **load_kw):

    if hor_avg:
        avg = "_avg_" + "".join(avg_dims)
    else:
        avg = ""

    # check if postprocessed output already exists
    if skip_exist:
        skip = True
    else:
        skip = False

    for outfile in outfiles:
        for var in variables:
            fpath = "{}/postprocessed/{}/{}{}.nc".format(outpath, var.upper(), outfile, avg)
            if os.path.isfile(fpath):
                if (not skip_exist) and (rank == 0):
                    os.remove(fpath)
            else:
                skip = False

    load_kw = dict(inst_file=inst_file, mean_file=mean_file, start_time=start_time,
                   pre_iloc=pre_iloc, pre_loc=pre_loc, **load_kw)
    kwargs = dict(budget_methods=budget_methods, t_avg=t_avg, t_avg_interval=t_avg_interval,
                  hor_avg=hor_avg, avg_dims=avg_dims, skip_exist=skip_exist,
                  save_output=save_output, return_model_output=return_model_output, **load_kw)

    if skip:
        print("Postprocessed output already available!")
        out = {var: load_postproc(outpath, var, avg=avg) for var in variables}
        if return_model_output:
            print("Load model output")
            dat_mean, dat_inst = load_data(outpath, **load_kw)
            out = [out, dat_inst, dat_mean]
        return out

    if chunks is not None:
        if any([c not in xy for c in chunks.keys()]):
            raise ValueError("Chunking is only allowed in the x and y-directions! "
                             "Given chunks: {}".format(chunks))
        if hor_avg:
            if any([d in avg_dims for d in chunks.keys()]):
                raise ValueError("Averaging dimensions cannot be used for chunking!")
        kwargs["return_model_output"] = False
        all_tiles = create_tasks(outpath, chunks=chunks, **load_kw)
        all_tiles = [(i, t) for i, t in enumerate(all_tiles)]
        tiles = all_tiles[rank::nproc]

        done = 0
        if len(tiles) == 0:
            done = 1
        if comm is not None:
            local_comm = comm.Split(done)
        else:
            local_comm = None

        for i, (task, tile) in enumerate(tiles):
            tile = {k: v for d in tile for k, v in d.items()}
            if tile == {}:
                tile = None
                task = None
            calc_tendencies_core(variables, outpath, tile=tile,
                                 task=task, comm=local_comm, **kwargs)
            # remove finished processors from communicator
            done = int(i == len(tiles) - 1)
            if comm is not None:
                local_comm = local_comm.Split(done)

        if comm is not None:
            comm.Barrier()
        if rank == 0:
            print("Load entire postprocessed output")
            out = {var: load_postproc(outpath, var, avg=avg) for var in variables}
            if return_model_output:
                dat_mean, dat_inst = load_data(outpath, **load_kw)
                out = [out, dat_inst, dat_mean]
            return out

    else:
        if nproc > 1:
            raise ValueError("Number of processors > 1, but chunking is disabled (chunks=None)!")
        return calc_tendencies_core(variables, outpath, **kwargs)


def calc_tendencies_core(variables, outpath, budget_methods="castesian correct",
                         tile=None, task=None, comm=None, t_avg=False, t_avg_interval=None,
                         hor_avg=False, avg_dims=None, skip_exist=True, save_output=True,
                         return_model_output=True, **load_kw):

    print("Load model output")
    dat_mean_all, dat_inst_all = load_data(outpath, **load_kw)
    dat_mean = dat_mean_all
    dat_inst = dat_inst_all

    # check if periodic bc can be used in staggering operations
    cyclic = {d: bool(dat_inst_all.attrs["PERIODIC_{}".format(d.upper())]) for d in xy}
    cyclic["bottom_top"] = False

    # select tile
    if tile is not None:
        print("\n\n{0}\nProcess tile: {1}\n".format("#" * 30, tile))
        dat_mean = dat_mean_all[tile]
        dat_inst = dat_inst_all[tile]
        # periodic BC cannot be used in tiling
        cyclic = {d: cyclic[d] and (d not in tile) for d in cyclic.keys()}

    if np.prod(list(dat_mean.sizes.values())) == 0:
        raise ValueError("At least one dimension is empy after indexing!")

    if hor_avg:
        avg = "_avg_" + "".join(avg_dims)
    else:
        avg = ""

    dat_mean, dat_inst, grid, attrs = prepare(dat_mean, dat_inst, variables=variables,
                                              cyclic=cyclic, t_avg=t_avg,
                                              t_avg_interval=t_avg_interval,
                                              hor_avg=hor_avg, avg_dims=avg_dims)
    datout_all = {}
    # prepare variable tendencies
    for var in variables:
        datout = {}
        VAR = var.upper()
        print("\n\n{0}\nProcess variable {1}\n".format("#" * 20, VAR))
        if skip_exist:
            # check if postprocessed output already exists
            skip = True
            for outfile in outfiles:
                fpath = "{}/postprocessed/{}/{}{}.nc".format(outpath, VAR, outfile, avg)
                if not os.path.isfile(fpath):
                    skip = False
            if skip:
                print("Postprocessed output already available!")
                datout_all[var] = load_postproc(outpath, var, avg=avg)
                continue

        dat_mean, dat_inst, sgs, sgsflux, sources, sources_sum, grid, dim_stag, mapfac, \
            = calc_tend_sources(dat_mean, dat_inst, var, grid, cyclic, attrs,
                                hor_avg=hor_avg, avg_dims=avg_dims)

        # calc fluxes and tendencies
        IDcs = []
        budget_methods = make_list(budget_methods)
        for comb in budget_methods:
            datout_c = {}
            c, comb, IDc = get_comb(comb.copy())
            IDcs.append(IDc)
            print("\n" + IDc)
            if c["dz_out"]:
                if not c["cartesian"]:
                    raise ValueError("dz_out can only be used for cartesian calculations!")
            elif c["corr_varz"]:
                raise ValueError("corr_varz can only be used together with dz_out!")

            # calculate total tendency
            total_tend = total_tendency(dat_inst, var, grid, dz_out=c["dz_out"],
                                        hor_avg=hor_avg, avg_dims=avg_dims, cyclic=cyclic, **attrs)

            dat = adv_tend(dat_mean, VAR, grid, mapfac, cyclic, attrs,
                           hor_avg=hor_avg, avg_dims=avg_dims,
                           cartesian=c["cartesian"], force_2nd_adv=c["force_2nd_adv"],
                           dz_out=c["dz_out"], corr_varz=c["corr_varz"])
            if dat is None:
                continue
            else:
                datout_c["flux"], datout_c["adv"], vmean, var_stag, corr = dat

            datout_c["tend"] = total_tend

            if c["correct"] and c["cartesian"]:
                out = cartesian_corrections(VAR, dim_stag, corr, var_stag, vmean,
                                            dat_mean["RHOD_MEAN"], grid, datout_c["adv"],
                                            total_tend, cyclic, dz_out=c["dz_out"],
                                            hor_avg=hor_avg, avg_dims=avg_dims)
                datout_c["adv"], datout_c["tend"], datout_c["corr"] = out
            # add all forcings
            datout_c["forcing"] = datout_c["adv"].sel(comp="adv_r", drop=True).sum("dir") + sources_sum

            if "dim" in datout_c["tend"].coords:
                datout_c["tend"] = datout_c["tend"].drop("dim")

            # aggregate different IDs
            loc = dict(ID=[IDc])
            for dn in datout_c.keys():
                datout_c[dn] = datout_c[dn].expand_dims(loc)
                if dn not in datout:
                    datout[dn] = datout_c[dn]
                else:
                    datout[dn] = xr.concat([datout[dn], datout_c[dn]], "ID")

        datout["tend"] = datout["tend"].expand_dims(comp=["tendency"])
        datout["forcing"] = datout["forcing"].expand_dims(comp=["forcing"])
        datout["tend"] = xr.concat([datout["tend"], datout["forcing"]], "comp")
        del datout["forcing"]
        adv_sum = datout["adv"].sum("dir")
        datout["adv"] = datout["adv"].reindex(dir=[*XYZ, "sum"])
        datout["adv"].loc[{"dir": "sum"}] = adv_sum

        # set units and descriptions
        units = units_dict_tend[var]
        units_flx = units_dict_flux[var]
        datout["sgsflux"] = sgsflux.assign_attrs(description="SGS {}-flux".format(VAR),
                                                 units=units_flx)
        datout["flux"] = datout["flux"].assign_attrs(description="resolved {}-flux".format(VAR),
                                                     units=units_flx)
        datout["adv"] = datout["adv"].assign_attrs(description="advective {}-tendency".format(VAR),
                                                   units=units)
        datout["sgs"] = sgs.assign_attrs(description="SGS {}-tendency".format(VAR), units=units)
        datout["tend"] = datout["tend"].assign_attrs(description="{}-tendency".format(VAR),
                                                     units=units)
        datout["sources"] = sources.assign_attrs(description="{}-tendency sources".format(VAR),
                                                 units=units)
        if c["correct"] and c["cartesian"]:
            datout["corr"] = datout["corr"].assign_attrs(
                description="{}-tendency correction".format(VAR), units=units)
        grid["MU_STAG_MEAN"] = grid["MU_STAG_MEAN"].assign_attrs(
            description="time-averaged dry air mass", units="Pa")
        datout["grid"] = grid

        if save_output:
            print("\nSave data")

        for dn, dat in datout.items():
            warn_duplicate_dim(dat, name=dn)

            # add height as coordinate
            if "flux" in dn:
                for D in XYZ:
                    z = stagger_like(grid["ZW"], dat[D], cyclic=cyclic, **grid[stagger_const])
                    z = z.assign_attrs(description=z.description + " staggered to {}-flux grid".format(D))
                    dat[D] = dat[D].assign_coords({"zf{}".format(D.lower()): z})
            elif dn != "grid":
                dat = dat.assign_coords(z=grid["Z_STAG"])

            da_type = False
            if type(dat) == DataArray:
                da_type = True
                dat = dat.to_dataset(name=dn)

            for v in dat.variables:
                dat[v].attrs = {k: v for k, v in dat[v].attrs.items() if k not in del_attrs}

            dat = dat.assign_attrs(attrs)
            if tile is not None:
                # strip tile boundary points
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
                            stop = -2
                        else:
                            stop = -1
                    t_bounds[d] = slice(start, stop)
                dat = dat[t_bounds]

            if save_output:
                fpath = "{}/postprocessed/{}/".format(outpath, VAR)
                os.makedirs(fpath, exist_ok=True)
                fpath += dn + avg + ".nc"
                if tile is None:
                    dat.to_netcdf(fpath)
                else:
                    save_tiles(dat, dn, fpath, dat_mean_all, task, tile, comm=comm)

            if da_type:
                dat = dat[dn]

            datout[dn] = dat

        if tile is None:
            datout_all[var] = datout

    if tile is None:
        out = datout_all
        if return_model_output:
            out = [out, dat_inst, dat_mean]
        return out


# %% tile processing

def create_tasks(outpath, chunks, **load_kw):
    print("Create tasks")
    dat_mean, dat_inst = load_data(outpath, **load_kw)
    tiles = []
    for dim, size in chunks.copy().items():
        if dim not in dat_mean.dims:
            raise ValueError("Chunking dimension {} not in data!".format(dim))
        bounds = np.arange(len(dat_mean[dim]))[::size]
        if len(bounds) == 1:
            print("Chunking in {0}-direction leads to one chunk only. "
                  "Deleting {0} from chunks dictionary.".format(dim))
            del chunks[dim]
            continue
        iloc = []
        for i in range(len(bounds)):
            iloc_b = {}
            for stag in [False, True]:
                if stag:
                    dim_s = dim + "_stag"
                    ext = 2
                else:
                    dim_s = dim
                    ext = 1
                if i == 0:
                    start = None
                else:
                    start = bounds[i] - 1
                if i == len(bounds) - 1:
                    stop = None
                else:
                    stop = bounds[i + 1] + ext

                iloc_b[dim_s] = slice(start, stop)
            iloc.append(iloc_b)
        tiles.append(iloc)
    tiles = list(itertools.product(*tiles))

    return tiles


def save_tiles(dat, name, fpath, dat_mean_all, task, tile, comm=None):
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
    coords_all = {d: dat_mean_all[d].values for d in tile.keys() if d in dat.dims}
    if mode == "w":
        tempfile = fpath + ".tmp"
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

