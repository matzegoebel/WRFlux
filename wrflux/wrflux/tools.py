#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 12:53:04 2019

@author: Matthias Göbel
"""
import numpy as np
import xarray as xr
xr.set_options(arithmetic_join="exact")
xr.set_options(keep_attrs=True)
import os
import pandas as pd
from datetime import datetime
from functools import partial
import socket

print = partial(print, flush=True)

dim_dict = dict(x="U",y="V",bottom_top="W",z="W")
xy = ["x", "y"]
XY = ["X", "Y"]
XYZ = [*XY, "Z"]
uvw = ["u", "v", "w"]
units_dict = {"t" : "K ", "q" : "", **{v : "ms$^{-1}$" for v in uvw}}
units_dict_tend = {"t" : "Ks$^{-1}$", "q" : "s$^{-1}$", **{v : "ms$^{-2}$" for v in uvw}}
units_dict_flux = {"t" : "Kms$^{-1}$", "q" : "ms$^{-1}$", **{v : "m$^{2}$s$^{-2}$" for v in uvw}}
units_dict_tend_rho = {"t" : "kg m$^{-3}$Ks$^{-1}$", "q" : "kg m$^{-3}$s$^{-1}$", **{v : "kg m$^{-2}$s$^{-2}$" for v in uvw}}
g = 9.81
rvovrd = 461.6/287.04
stagger_const = ["FNP", "FNM", "CF1", "CF2", "CF3", "CFN", "CFN1"]

outfiles = ["grid", "adv", "flux", "tend", "sources", "sgs", "sgsflux", "corr"]
# outfiles = ["grid", "adv", "flux", "tend", "sources", "sgs", "sgsflux"]
#%% figloc
host = socket.gethostname()
basedir = "~/phd/"
basedir = os.path.expanduser(basedir)
figloc = basedir + "figures/"

#%%open dataset
def fix_coords(data, dx, dy):
    """Assign time and space coordinates"""

    #assign time coordinate
    if ("XTIME" in data) and (type(data.XTIME.values[0]) == np.datetime64):
        data = data.assign_coords(Time=data.XTIME)
    else:
        time = data.Times.astype(str).values
        time = pd.DatetimeIndex([datetime.fromisoformat(str(t)) for t in time])
        data = data.assign_coords(Time=time)

    for v in ["XTIME", "Times"]:
        if v in data:
            data = data.drop(v)
    #assign x and y coordinates and rename dimensions
    for dim_old, res, dim_new in zip(["south_north", "west_east"], [dy, dx], ["y", "x"]):
        for stag in [False, True]:
            if stag:
                dim_old = dim_old + "_stag"
                dim_new = dim_new + "_stag"
            if dim_old in data.dims:
                data[dim_old] = np.arange(data.sizes[dim_old]) * res
                data[dim_old] = data[dim_old] - (data[dim_old][-1] + res)/2
                data[dim_old] = data[dim_old].assign_attrs({"units" : "m"})
                data = data.rename({dim_old : dim_new})

    #assign vertical coordinate
    if ("ZNW" in data) and ("bottom_top_stag" in data.dims):
        data = data.assign_coords(bottom_top_stag=data["ZNW"].isel(Time=0,drop=True))
    if ("ZNU" in data) and ("bottom_top" in data.dims):
        data = data.assign_coords(bottom_top=data["ZNU"].isel(Time=0,drop=True))

    return data

def open_dataset(file, var=None, chunks=None, del_attrs=True, fix_c=True, **kwargs):
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

    if var is not None:
        var = make_list(var)
        ds = ds[var]

    if chunks is not None:
        ds = chunk_data(ds, chunks)

    if del_attrs:
        ds.attrs = {}

    return ds

def chunk_data(data, chunks):
    chunks = {d : c for d,c in chunks.items() if d in data.dims}
    return data.chunk(chunks)

#%%misc



def make_list(o):
    if type(o) not in [tuple, list, dict, np.ndarray]:
        o = [o]
    return o

def time_avg(dat_mean, cyclic, stagger_kw, avg_kwargs):
    exclude = ["CORR", "TEND", "RHOD_MEAN", "MUT_MEAN", "WW_MEAN", "_VAR"]
    out = xr.Dataset()
    rho = dat_mean["RHOD_MEAN"]
    for var in dat_mean.data_vars:
        if ("_MEAN" in var) and (var != "Z_MEAN") and all([e not in var for e in exclude]):
            rho_s = stagger_like(rho, dat_mean[var], cyclic=cyclic, **stagger_kw)
            rho_s_mean = rho_s.coarsen(**avg_kwargs).mean()
            out[var] = (rho_s*dat_mean[var]).coarsen(**avg_kwargs).mean()/rho_s_mean
        else:
            out[var] = dat_mean[var].coarsen(**avg_kwargs).mean()

    return out

def avg_xy(data, avg_dims, attrs=None, rho=None, cyclic=None, **stagger_const):
    """Average data over the given dimensions even
    if the actual present dimension has '_stag' added.
    If rho is given, do density-weighted averaging.
    Before averaging, cut right boundary points for periodic BC
    and both boundary points for non-periodic BC."""

    if type(data) == xr.core.dataset.Dataset:
        out = xr.Dataset()
        for v in data.data_vars:
            out[v] = avg_xy(data[v], avg_dims, attrs=attrs, rho=rho, cyclic=cyclic, **stagger_const)
        return out

    avg_dims_final = avg_dims.copy()
    if rho is not None:
        rho_s = stagger_like(rho, data, cyclic=cyclic, **stagger_const)
        rho_s_mean = avg_xy(rho_s, avg_dims, attrs=attrs)

    for i,d in enumerate(avg_dims):
        ds = d +  "_stag"
        if ds in data.dims:
            avg_dims_final.append(ds)
        if d not in data.dims:
            avg_dims_final.remove(d)

        #cut boundary points depending on lateral BC
        if (attrs is not None) and (not attrs["PERIODIC_{}".format(d.upper())]):
            data = loc_data(data,iloc={d : slice(1,-1)})
            if rho is not None:
                rho_s = loc_data(rho_s,iloc={d : slice(1,-1)})
        elif ds in data.dims:
            data = data[{ds : slice(0,-1)}]
            if rho is not None:
                rho_s = rho_s[{ds : slice(0,-1)}]

    if rho is None:
        return data.mean(avg_dims_final)
    else:
        return (rho_s*data).mean(avg_dims_final)/rho_s_mean

def find_bad(dat, nan=True, inf=True):
    """Drop all indeces of each dimension that do not contain NaNs or infs."""

    nans = False
    infs = False
    if nan:
        nans = dat.isnull()
    if inf:
        infs = dat == np.inf
    invalid = nans | infs
    invalid = invalid.where(invalid)
    invalid = dropna_dims(invalid)
    dat = dat.loc[invalid.indexes]

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
    err = err.max(dim=dim)/norm
    if err.shape != ():
        err = err.max()
    return float(err)

def index_of_agreement(dat, ref, dim=None):
    """
    Index of agreement by Willmott (1981)

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
        index of agreement.

    """

    if dim is not None:
        dim = correct_dims_stag_list(dim, ref)

    mse = ((dat-ref)**2).mean(dim=dim)
    ref_mean = ref.mean(dim=dim)
    norm = ((abs(dat -ref_mean) + abs(ref -ref_mean))**2).mean(dim=dim)
    return 1 - mse/norm

def nse(dat, ref, dim=None):
    """
    Nash–Sutcliffe model efficiency coefficient

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
        dim = correct_dims_stag_list(dim, ref)

    mse = ((dat-ref)**2).mean(dim=dim)
    norm = ((ref - ref.mean(dim=dim))**2).mean(dim=dim)
    return 1 - mse/norm

def warn_duplicate_dim(data, name=None):
    """Warn if dataarray or dataset contains the staggered and unstaggered version of any dimension"""
    if type(data) == xr.core.dataset.Dataset:
        for v in data.data_vars:
            warn_duplicate_dim(data[v], name=v)
        return

    if name is None:
        name = data.name
    for d in data.dims:
        if d + "_stag" in data.dims:
            print("WARNING: datarray {0} contains both dimensions {1} and {1}_stag".format(name, d))

def rolling_mean(ds, dim, window, periodic=True, center=True):
    if periodic:
        if center:
            pad = int(np.floor(window/2))
            pad = (pad, pad)
        else:
            pad = (window - 1, 0)
        ds = ds.pad({dim : pad}, mode='wrap')
    ds = ds.rolling({dim : window}, center=center).mean()
    if periodic:
        ds = ds.isel({dim : np.arange(pad[0], len(ds[dim]) - pad[1])})
    return ds

def correct_dims_stag(loc, dat):
    """
    Add "_stag" to every key in dictionary loc, for which the unmodified key
    is not a dimension of dat but the modified key is. Delete keys that are
    not dimensions of dat. Returns a copy.

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
        if d not in dat.dims:
            if d + "_stag" in dat.dims:
                loc_out[d + "_stag"] = val
            del loc_out[d]
    return loc_out

def correct_dims_stag_list(l, dat):
    """
    Add "_stag" to every item in iterable l, for which the unmodified item
    is not a dimension of dat but the modified item is. Delete items that are
    not dimensions of dat. Returns a copy.

    Parameters
    ----------
    l : iterable
        input iterable.
    dat : datarray or dataset
        reference data.

    Returns
    -------
    loc : list
        modified list.

    """

    l_out = []
    for i,d in enumerate(l):
        if d in dat.dims:
            l_out.append(d)
        else:
            if d + "_stag" in dat.dims:
                l_out.append(d + "_stag")
    return l_out

def loc_data(dat, loc=None, iloc=None, copy=True):

    if type(dat) == xr.core.dataset.Dataset:
        out = xr.Dataset()
        for v in dat.data_vars:
            out[v] = loc_data(dat[v], loc=loc, iloc=iloc, copy=copy)
        return out

    if copy:
        dat = dat.copy()
    if iloc is not None:
        iloc = correct_dims_stag(iloc, dat)
        dat = dat[iloc]
    if loc is not None:
        loc = correct_dims_stag(loc, dat)
        dat = dat.loc[loc]

    return dat

#%%manipulate datasets

def select_ind(a, axis=0, indeces=0):
    """Select indeces along (possibly multiple) axis"""
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
    cyclic : dict of all dims or None, optional
        use periodic boundary conditions to fill lateral boundary points. The default is False.
    ignore : list, optional
        dimensions to ignore
    **stagger_kw : dict
        keyword arguments for staggering.

    Returns
    -------
    data : xarray dataarray or dataset
        output data.

    """

    if type(data) == xr.core.dataset.Dataset:
        out = xr.Dataset()
        for v in data.data_vars:
            out[v] = stagger_like(data[v], ref, rename=rename, cyclic=cyclic, ignore=ignore, **stagger_kw)
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
        use periodic boundary conditions to fill lateral boundary points. The default is False.
    **interp_const : dict
        vertical extrapolation constants

    Returns
    -------
    data_stag : xarray dataarray
        staggered data.

    """

    if dim == "bottom_top":
        data_stag = data*FNM + data.shift({dim : 1})*FNP
    else:
        data_stag = 0.5*(data + data.roll({dim : 1}, roll_coords=False))

    data_stag = post_stagger(data_stag, dim, new_coord, rename=rename, data=data, cyclic=cyclic, **interp_const)

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
        use periodic boundary conditions to fill lateral boundary points. The default is False.
    **interp_const : dict
        vertical extrapolation constants
.

    Returns
    -------
    data_stag : xarray dataarray
        staggered data.
    """
    dim_s = dim
    if rename:
        dim_s = dim + "_stag"
        data_stag = data_stag.rename({dim : dim_s})

    data_stag[dim_s] = new_coord[:-1]
    data_stag = data_stag.reindex({dim_s : new_coord})

    c = new_coord

    #fill boundary values
    if dim == "bottom_top":
        if interp_const != {}:
            data_stag[{dim_s: 0}] = interp_const["CF1"]*data[{dim : 0}] + interp_const["CF2"]*data[{dim : 1}] + interp_const["CF3"]*data[{dim : 2}]
            data_stag[{dim_s : -1}] = interp_const["CFN"]*data[{dim : -1}] + interp_const["CFN1"]*data[{dim : -2}]
    elif cyclic:
        #set second boundary point equal to first
        data_stag.loc[{dim_s : c[-1]}] = data_stag.loc[{dim_s : c[0]}]
    else:
        #also set first boundary point to NaN
        data_stag.loc[{dim_s : c[0]}] = np.nan

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

    data = 0.5*(data + data.shift({dim : -1}))
    data = data.sel({dim:data[dim][:-1]})
    new_dim = dim
    if rename:
        new_dim = dim[:dim.index("_stag")]
        data = data.rename({dim : new_dim})

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
        data_s = data.shift({dim : 1})
    else:
        data_s = data.roll({dim : 1}, roll_coords=False)

    out = data - data_s
    if "_stag" in dim:
        out = out.sel({dim : out[dim][1:]})
        new_dim = dim
        if rename and (dim != "Time"):
            new_dim = dim[:dim.index("_stag")]
            out = out.rename({dim : new_dim})
        out[new_dim] = new_coord
    else:
        out = post_stagger(out, dim, new_coord, rename=rename, cyclic=cyclic)

    return out

def remove_deprecated_dims(ds):
    """Remove dimensions that do not occur in any of the variables of the given dataset"""
    var_dims = []
    for v in ds.data_vars:
        var_dims.extend(ds[v].dims)

    for d in ds.dims:
        if d not in var_dims:
            ds = ds.drop(d)
    return ds
#%% WRF tendencies

def sgs_tendency(dat_mean, VAR, grid, dzdd, cyclic, dim_stag=None, mapfac=None):
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
    sgs["Z"] = -diff(fz*rhoz, d3s, new_coord=vcoord)
    sgs["Z"] = sgs["Z"]/dn/grid["MU_STAG_MEAN"]*(-g)
    for d, v in zip(xy, ["U", "V"]):
        #compute corrections
        du = d.upper()
        if mapfac is None:
            m = 1
        else:
            m = mapfac[du]
        fd = dat_mean["F{}{}_SGS_MEAN".format(VAR, du)]
        sgsflux[du] = fd
        fd = fd*stagger_like(dat_mean["RHOD_MEAN"], fd, cyclic=cyclic, **grid[stagger_const])
        cyc = cyclic[d]
        if d in fd.dims:
            #for momentum variances
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
            flux8z[:,[0,-1]] = 0
        corr_sgs = diff(flux8z, d3s, new_coord=vcoord)/dn
        corr_sgs = corr_sgs*stagger_like(dzdd[du], corr_sgs, cyclic=cyclic, **grid[stagger_const])

        dx = grid["D" + du]
        sgs[du] = -diff(fd, ds, new_coord=sgs[d], cyclic=cyc)/dx*m
        if VAR == "W":
            m = mapfac["Y"]
        sgs[du] = sgs[du]/grid["RHOD_STAG_MEAN"] + corr_sgs*m/grid["MU_STAG_MEAN"]*(-g)

    sgsflux["Z"] = fz
    sgs = sgs[XYZ]
    sgs = sgs.to_array("dir")
    if VAR == "W":
        sgs[:,:,[0,-1]] = 0

    return sgs, sgsflux


def adv_tend(dat_mean, VAR, grid, mapfac, cyclic, attrs, hor_avg=False, avg_dims=None,
             cartesian=False, force_2nd_adv=False, dz_out=False, corr_varz=False):

    print("Compute resolved tendencies")

    #get appropriate staggered variables, vertical velocity, and flux variables
    var_stag = xr.Dataset()
    fluxnames = ["F{}{}_ADV_MEAN".format(VAR, d) for d in XYZ]
    if force_2nd_adv:
        fluxnames = [fn + "_2ND" for fn in fluxnames]
        for d, f in zip(XYZ, fluxnames):
            var_stag[d] = stagger_like(dat_mean["{}_MEAN".format(VAR)], dat_mean[f], cyclic=cyclic, **grid[stagger_const])
    else:
        for d in XYZ:
            var_stag[d] = dat_mean["{}{}_MEAN".format(VAR, d)]

    if cartesian:
        w = dat_mean["WD_MEAN"]
    else:
        w = dat_mean["OMZN_MEAN"]

    print("fluxes: " + str(fluxnames))
    if not all([f in dat_mean for f in fluxnames]):
        raise ValueError("Fluxes not available!")

    vmean = xr.Dataset({"X" : dat_mean["U_MEAN"], "Y" : dat_mean["V_MEAN"], "Z" : w})
    if hor_avg:
        var_stag = avg_xy(var_stag, avg_dims, attrs, rho=dat_mean["RHOD_MEAN"], cyclic=cyclic, **grid[stagger_const])
        for k in vmean.keys():
            vmean[k] = avg_xy(vmean[k], avg_dims, attrs, rho=dat_mean["RHOD_MEAN"], cyclic=cyclic, **grid[stagger_const])

    tot_flux = dat_mean[fluxnames]
    tot_flux = tot_flux.rename(dict(zip(fluxnames, XYZ)))
    rhod8z = stagger_like(dat_mean["RHOD_MEAN"], tot_flux["Z"], cyclic=cyclic, **grid[stagger_const])


    corr = ["F{}X_CORR".format(VAR), "F{}Y_CORR".format(VAR), "CORR_D{}DT".format(VAR)]
    if force_2nd_adv:
        corr = [corri + "_2ND" for corri in corr]
    corr = dat_mean[corr]
    corr = corr.to_array("dir")
    corr["dir"] = ["X", "Y", "T"]

    if not cartesian:
        tot_flux["Z"] = tot_flux["Z"] - (corr.loc["X"] + corr.loc["Y"] + corr.loc["T"])/rhod8z

    if dz_out:
        if corr_varz:
            corr.loc["X"] = dat_mean["F{}X_CORR_DZOUT".format(VAR)]
            corr.loc["Y"] = dat_mean["F{}Y_CORR_DZOUT".format(VAR)]
            corr_t = dat_mean[VAR + "_MEAN"]
            corr_t = rhod8z*stagger_like(corr_t, rhod8z, cyclic=cyclic, **grid[stagger_const])
            corr["T"] = corr_t
        else:
            corr = tot_flux[XY]
            corr["T"] = dat_mean[VAR + "_MEAN"]
            corr = rhod8z*stagger_like(corr, rhod8z, cyclic=cyclic, **grid[stagger_const])
            corr = corr.to_array("dir")

  #  mean advective fluxes
    mean_flux = xr.Dataset()
    for d in XYZ:
        if hor_avg and (d.lower() in avg_dims):
            mean_flux[d] = 0.
        else:
            vel_stag = stagger_like(vmean[d], ref=var_stag[d], cyclic=cyclic, **grid[stagger_const])
            if (VAR == "W") and (d in XY):
                vel_stag[{"bottom_top_stag" : 0}] = 0
            mean_flux[d] = var_stag[d]*vel_stag

    #advective tendency from fluxes
    adv = {}
    fluxes = {"adv_r" : tot_flux, "mean" : mean_flux}
    try:
        #explicit resolved turbulent fluxes
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
        if (comp in ["trb_r", "mean"]) and hor_avg: #need trb_r?
            mf = avg_xy(mapfac, avg_dims)
            rhod8z_m = avg_xy(rhod8z, avg_dims)
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
            if (comp in ["trb_r", "mean"]) and hor_avg:#TODOm: correct?
                mf_flx = avg_xy(mf_flx, avg_dims)
                fac = avg_xy(fac, avg_dims)
            if not dz_out:
                fac = build_mu(fac, grid, full_levels="bottom_top_stag" in flux[du].dims)
            fac = stagger_like(fac, flux[du], cyclic=cyclic, **grid[stagger_const])
            adv_i[du] = -diff(fac*flux[du]/mf_flx, ds, dat_mean[d], cyclic=cyc)*mf["X"]*mf["Y"]/dx
        fz = rhod8z_m*flux["Z"]
        if VAR == "W":
            adv_i["Z"] = -diff(fz, "bottom_top", grid["ZNW"])/grid["DN"]
            #set sfc and top point correctly
            adv_i["Z"][{"bottom_top_stag" : 0}] = 0.
            adv_i["Z"][{"bottom_top_stag" : -1}] = (2*fz.isel(bottom_top=-1)/grid["DN"][-2]).values

        else:
            adv_i["Z"] = -diff(fz, "bottom_top_stag", grid["ZNU"])/grid["DNW"]
        adv_i["Z"] = adv_i["Z"]*(-g)
        for d in adv_i.data_vars:
            if dz_out and (d != "Z"):
                adv_i[d] = adv_i[d]/grid["RHOD_STAG_MEAN"]
            else:
                adv_i[d] = adv_i[d]/grid["MU_STAG_MEAN"]

        adv[comp] = adv_i

    if hor_avg:
        adv["adv_r"] = avg_xy(adv["adv_r"], avg_dims, attrs)
        fluxes["adv_r"] = avg_xy(fluxes["adv_r"], avg_dims, attrs)

    keys = adv.keys()
    adv = xr.concat(adv.values(), "comp")
    adv = adv.to_array("dir")
    adv["comp"] = list(keys)
    flux = xr.concat(fluxes.values(), "comp")
    flux["comp"] = list(fluxes.keys())

    #resolved turbulent fluxes and tendencies
    if not trb_exp:
        flux = flux.reindex(comp=["adv_r", "mean", "trb_r"])
        adv = adv.reindex(comp=["adv_r", "mean", "trb_r"])
        for d in flux.data_vars:
            flux[d].loc["trb_r"] = flux[d].loc["adv_r"] - flux[d].loc["mean"]
            adv.loc[d, "trb_r"] = adv.loc[d, "adv_r"] - adv.loc[d, "mean"]
    elif hor_avg:
        #assign fluxes in averaged dimension to turbulent component
        for d in avg_dims:
            d = d.upper()
            flux[d].loc["trb_r"] = flux[d].loc["adv_r"] - flux[d].loc["mean"]
            adv.loc[d, "trb_r"] = adv.loc[d, "adv_r"] - adv.loc[d, "mean"]

    return flux, adv, vmean, var_stag, corr

def cartesian_corrections(VAR, dim_stag, corr, var_stag, vmean, rhodm, grid, mapfac, adv, tend,
                          cyclic, attrs, dz_out=False, hor_avg=False, avg_dims=None):

    print("Compute Cartesian corrections")
    #decompose cartesian corrections
    #total
    corr = corr.expand_dims(comp=["adv_r"]).reindex(comp=["adv_r", "mean", "trb_r"])
    if hor_avg:
        corr = avg_xy(corr, avg_dims, attrs)
        rhodm = avg_xy(rhodm, avg_dims)

    #mean part
    kw = dict(ref=var_stag["Z"], cyclic=cyclic, **grid[stagger_const])
    rho_stag =  stagger_like(rhodm, **kw)
    for d, v in zip(xy, ["U", "V"]):
        #staggering
        du = d.upper()
        if hor_avg and (d in avg_dims):
            corr.loc["mean", du] = 0
            continue

        if dz_out:
            corr_d = stagger_like(vmean[du], **kw)
        else:
            corr_d = stagger_like(grid["dzd{}_{}".format(d, v.lower())], **kw)
        corr.loc["mean", du] = corr_d*rho_stag*var_stag["Z"]

    dzdt = stagger_like(grid["dzdd"].loc["T"], **kw)
    corr.loc["mean", "T"] = rho_stag*dzdt*var_stag["Z"]

    #resolved turbulent part
    corr.loc["trb_r"] = corr.loc["adv_r"] - corr.loc["mean"]

    #correction flux to tendency
    if "W" in VAR:
        dcorr_dz = diff(corr, "bottom_top", grid["ZNW"])/grid["DN"]
        dcorr_dz[{"bottom_top_stag" : 0}] = 0.
        dcorr_dz[{"bottom_top_stag" : -1}] = -(2*corr.isel(bottom_top=-1)/grid["DN"][-2]).values
    else:
        dcorr_dz = diff(corr, "bottom_top_stag", grid["ZNU"])/grid["DNW"]
    dcorr_dz = dcorr_dz*(-g)/grid["MU_STAG_MEAN"]

    if dz_out:
        dcorr_dz = dcorr_dz*stagger_like(grid["dzdd"], dcorr_dz, cyclic=cyclic, **grid[stagger_const])

    #apply corrections
    for i, d in enumerate(XY):
        adv.loc[d] = adv.loc[d] + dcorr_dz[:, i]

    tend = tend - dcorr_dz.sel(comp="adv_r", dir="T", drop=True)

    return adv, tend, dcorr_dz

def total_tendency(dat_inst, var, grid, dz_out=False, hor_avg=False, avg_dims=None, **attrs):
    #instantaneous variable
    if var == "t":
        if attrs["USE_THETA_M"] and (not attrs["OUTPUT_DRY_THETA_FLUXES"]):
            if "THM" in dat_inst:
                vard = dat_inst["THM"]
            else:
                vard = (dat_inst["T"] + 300)*(1 + rvovrd*dat_inst["QVAPOR"]) - 300
        else:
            vard = dat_inst["T"]
    elif var == "q":
        vard = dat_inst["QVAPOR"]
    else:
        vard = dat_inst[var.upper()]

    #couple variable to mu
    if dz_out:
        rvar = vard*dat_inst["RHOD_STAG"]
    else:
        rvar = vard*dat_inst["MU_STAG"]

    # total tendency
    dt = int(dat_inst.Time[1] - dat_inst.Time[0])*1e-9
    total_tend = rvar.diff("Time")/dt

    if hor_avg:
        total_tend = avg_xy(total_tend, avg_dims)

    if dz_out:
        total_tend = total_tend/grid["RHOD_STAG_MEAN"]
    else:
        total_tend = total_tend/grid["MU_STAG_MEAN"]

    return total_tend

def load_postproc(outpath, var, avg=None):
    datout = {}
    if avg is None:
        avg = ""
    outpath = os.path.join(outpath, "postprocessed", var.upper())
    for f in outfiles:
        file = "{}/{}{}.nc".format(outpath, f, avg)
        if f in ["sgsflux","flux","grid"]:
           datout[f] = xr.open_dataset(file)
        else:
           datout[f] = xr.open_dataarray(file)
    return datout

def calc_tendencies(variables, outpath, inst_file=None, mean_file=None, start_time=None, budget_methods="castesian correct",
                    pre_iloc=None, pre_loc=None, t_avg=False, t_avg_interval=None, hor_avg=False, avg_dims=None, skip_exist=True,
                    save_output=True, return_model_output=True):

    if hor_avg:
        avg = "_avg_" + "".join(avg_dims)
    else:
        avg = ""

    skip = False
    if skip_exist:
        #check if postprocessed output already exists
        skip = True
        for c in outfiles:
            for var in variables:
                fpath = "{}/postprocessed/{}/{}{}.nc".format(outpath, var.upper(), c, avg)
                if not os.path.isfile(fpath):
                    skip = False

    if return_model_output or (not skip):
        budget_methods = make_list(budget_methods)
        dat = load_data(outpath, inst_file=inst_file, mean_file=mean_file, start_time=start_time,
                        pre_iloc=pre_iloc, pre_loc=pre_loc)
        if dat is None:
            return
        else:
            dat_mean, dat_inst = dat

    if skip:
         print("Postprocessed output already available!")
         out = {var : load_postproc(outpath, var, avg=avg) for var in variables}
         if return_model_output:
             out = [out, dat_inst, dat_mean]
         return out

    dat_mean, dat_inst, grid, cyclic, attrs = prepare(dat_mean, dat_inst, variables=variables,
                                                      t_avg=t_avg, t_avg_interval=t_avg_interval, hor_avg=hor_avg, avg_dims=avg_dims)
    datout_all = {}
    #prepare variable tendencies
    for var in variables:
        datout = {}
        VAR = var.upper()
        print("\n\n{0}\nProcess variable {1}\n".format("#"*20, VAR))
        if skip_exist:
            #check if postprocessed output already exists
            skip = True
            for c in outfiles:
                fpath = "{}/postprocessed/{}/{}{}.nc".format(outpath, VAR, c, avg)
                if not os.path.isfile(fpath):
                    skip = False
            if skip:
                print("Postprocessed output already available!")
                datout_all[var] = load_postproc(outpath, var, avg=avg)
                continue

        dat_mean, dat_inst, sgs, sgsflux, sources, sources_sum, grid, dim_stag, mapfac, \
         = calc_tend_sources(dat_mean, dat_inst, var, grid, cyclic, attrs, hor_avg=hor_avg, avg_dims=avg_dims)

        #calc fluxes and tendencies
        keys = ["cartesian","correct","dz_out","force_2nd_adv","corr_varz"] #available settings
        short_names = {"2nd" : "force_2nd_adv", "corr" : "correct"} #abbreviations for settings

        IDcs = []

        for comb in budget_methods:
            datout_c = {}
            c, comb, IDc = get_comb(comb.copy(), keys, short_names)
            IDcs.append(IDc)
            print("\n" + IDc)
            if c["dz_out"]:
                if not c["cartesian"]:
                    raise ValueError("dz_out can only be used for cartesian calculations!")
            elif c["corr_varz"]:
                raise ValueError("corr_varz can only be used together with dz_out!")

            #calculate total tendency
            total_tend = total_tendency(dat_inst, var, grid, dz_out=c["dz_out"], hor_avg=hor_avg, avg_dims=avg_dims, **attrs)

            dat = adv_tend(dat_mean, VAR, grid, mapfac, cyclic, attrs, hor_avg=hor_avg, avg_dims=avg_dims,
                                  cartesian=c["cartesian"], force_2nd_adv=c["force_2nd_adv"],
                                  dz_out=c["dz_out"], corr_varz=c["corr_varz"])
            if dat is None:
                continue
            else:
                datout_c["flux"], datout_c["adv"], vmean, var_stag, corr = dat

            datout_c["tend"] = total_tend

            if c["correct"] and c["cartesian"]:
                datout_c["adv"], datout_c["tend"], datout_c["corr"] = cartesian_corrections(VAR, dim_stag, corr, var_stag, vmean, dat_mean["RHOD_MEAN"],
                                                            grid, mapfac, datout_c["adv"], total_tend, cyclic, attrs, dz_out=c["dz_out"],
                                                            hor_avg=hor_avg, avg_dims=avg_dims)

            #add all forcings
            datout_c["forcing"] = datout_c["adv"].sel(comp="adv_r", drop=True).sum("dir") + sources_sum

            if "dim" in datout_c["tend"].coords:
                datout_c["tend"] = datout_c["tend"].drop("dim")

            #aggregate different IDs
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
        adv_sum = datout["adv"].sum("dir")
        datout["adv"] = datout["adv"].reindex(dir=[*XYZ, "sum"])
        datout["adv"].loc[{"dir":"sum"}] = adv_sum

        units = units_dict_tend[var]
        units_flx = units_dict_flux[var]
        datout["sgsflux"] = sgsflux.assign_attrs(description="SGS {}-flux".format(VAR), units=units_flx)
        datout["flux"] = datout["flux"].assign_attrs(description="resolved {}-flux".format(VAR), units=units_flx)
        datout["adv"] = datout["adv"].assign_attrs(description="advective {}-tendency".format(VAR), units=units)
        datout["sgs"] = sgs.assign_attrs(description="SGS {}-tendency".format(VAR), units=units)
        datout["tend" ]= datout["tend"].assign_attrs(description="{}-tendency".format(VAR), units=units)
        datout["sources"] = sources.assign_attrs(description="{}-tendency".format(VAR), units=units)
        if c["correct"] and c["cartesian"]:
            datout["corr"] = datout["corr"].assign_attrs(description="{}-tendency correction".format(VAR), units=units)
        desc = " staggered to {}-grid".format(VAR)
        grid["MU_STAG_MEAN"] = grid["MU_STAG_MEAN"].assign_attrs(description="time-averaged dry air mass" + desc, units="Pa")
        grid["Z_STAG"] = grid["Z_STAG"].assign_attrs(description="time-averaged geopotential height" + desc)
        datout["grid"] = grid

        for dn, d in datout.items():
            warn_duplicate_dim(d, name=dn)


        #save data
        print("\nSave data")
        if save_output:
            for dn, d in datout.items():
                #add height as coordinate
                if "flux" in dn:
                    for D in XYZ:
                        z = stagger_like(grid["ZW"], d[D], cyclic=cyclic, **grid[stagger_const])
                        z = z.assign_attrs(description=z.description +  " staggered to {}-flux grid".format(D))
                        d[D] = d[D].assign_coords({"zf{}".format(D.lower()) : z})
                elif dn != "grid":
                    d = d.assign_coords(z=grid["Z_STAG"])
                if "rep" in d.dims:
                    d = d.isel(rep=0, drop=True)
                d = d.assign_attrs(attrs)
                datout[dn] = d
                fpath = "{}/postprocessed/{}/{}{}.nc".format(outpath, VAR, dn, avg)
                if os.path.isfile(fpath):
                    os.remove(fpath)
                d.to_netcdf(fpath)
        datout_all[var] = datout

    out = datout_all
    if return_model_output:
        out = [out, dat_inst, dat_mean]
    return out

#%%prepare

def load_data(outpath, inst_file=None, mean_file=None, start_time=None, pre_iloc=None, pre_loc=None):
    dat_inst = []
    dat_mean = []
    print("Load files")
    if inst_file is None:
        if start_time is None:
            raise ValueError("Either inst_file or start_time must be given!")
        inst_file = "instout_d01_" + start_time
    if mean_file is None:
        if start_time is None:
            raise ValueError("Either mean_file or start_time must be given!")
        mean_file = "meanout_d01_" + start_time
    fpath = outpath + "/"

    dat_inst = open_dataset(fpath + inst_file, del_attrs=False)
    dat_mean = open_dataset(fpath + mean_file)

    #select subset of data
    if pre_iloc is not None:
        if "Time" in pre_iloc:
            raise ValueError("Time axis should not be indexed with iloc, as mean and inst output may have different frequencies!")
        dat_mean = dat_mean[pre_iloc]
        dat_inst = dat_inst[pre_iloc]
    if pre_loc is not None:
        dat_mean = dat_mean.loc[pre_loc]
        dat_inst = dat_inst.loc[pre_loc]
    if np.prod(list(dat_mean.sizes.values())) == 0:
        raise ValueError("At least one dimension is empy after indexing!")

    if len(dat_inst) * len(dat_mean) == 0:
        print("Inst or mean file not available at {}!".format(fpath))
        return

    dims = ["Time", "bottom_top", "bottom_top_stag", "soil_layers_stag", "y", "y_stag", "x", "x_stag", "seed_dim_stag", "rep"]
    dat_mean = dat_mean.transpose(*[d for d in dims if d in dat_mean.dims])
    dat_inst = dat_inst.transpose(*[d for d in dims if d in dat_inst.dims])

    return dat_mean, dat_inst


def get_comb(comb, keys, short_names):
    if (len(comb) > 0) and (type(comb[0]) == list):
        IDc = comb[1]
        comb = comb[0]
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

def prepare(dat_mean, dat_inst, variables=None, t_avg=False, t_avg_interval=None, hor_avg=False, avg_dims=None):
    print("Prepare data")
    attrs = dat_inst.attrs
    dat_inst.attrs = {}
    #check if periodic bc can be used in staggering operations
    cyclic = {d : bool(attrs["PERIODIC_{}".format(d.upper())]) for d in xy}
    cyclic["bottom_top"] = False

    #strip first time as dat_inst needs to be one time stamp longer
    dat_mean = dat_mean.sel(Time=dat_mean.Time[1:])
    if len(dat_mean.Time) == 0:
        raise ValueError("dat_mean is empty! Needs to contain at least two timesteps initially!")

    #computational grid
    grid = dat_inst[["ZNU","ZNW","DNW","DN","C1H","C2H","C1F","C2F",*stagger_const]].isel(Time=0, drop=True)
    grid["DN"] = grid["DN"].rename(bottom_top="bottom_top_stag").assign_coords(bottom_top_stag=grid["ZNW"][:-1]).reindex(bottom_top_stag=grid["ZNW"])
    grid["DX"] = attrs["DX"]
    grid["DY"] = attrs["DY"]

    dat_mean = dat_mean.assign_coords(bottom_top=grid["ZNU"], bottom_top_stag=grid["ZNW"])
    dat_inst = dat_inst.assign_coords(bottom_top=grid["ZNU"], bottom_top_stag=grid["ZNW"])

    dat_mean = dat_mean.rename(ZWIND_MEAN="W_MEAN")
    rhod = stagger(dat_mean["RHOD_MEAN"], "bottom_top", dat_mean["bottom_top_stag"], **grid[stagger_const])
    dat_mean["OMZN_MEAN"] = dat_mean["WW_MEAN"]/(-g*rhod)

    if t_avg:
        inst = dat_mean.copy()
        print("Average dat_mean over {} output steps".format(t_avg_interval))
        avg_kwargs = dict(Time=t_avg_interval, coord_func={"Time" : partial(select_ind, indeces=-1)}, boundary="trim")
        dat_mean = time_avg(dat_mean, cyclic=cyclic, stagger_kw=grid[stagger_const], avg_kwargs=avg_kwargs)

        #compute resolved turbulent fluxes explicitly if output contains all timesteps
        dt_out = float(inst.Time[1] - inst.Time[0])/1e9
        if round(dt_out) == attrs["DT"]:
            print("Compute turbulent fluxes explicitly")
            trb_fluxes(dat_mean, inst, variables, grid, cyclic, avg_kwargs, attrs,
                       hor_avg=hor_avg, avg_dims=avg_dims)


    #select start and end points of averaging intervals
    dat_inst = dat_inst.sel(Time=[dat_inst.Time[0].values, *dat_mean.Time.values])
    for v in dat_inst.coords:
        if ("XLAT" in v) or ("XLONG" in v):
            dat_inst = dat_inst.drop(v)

    return dat_mean, dat_inst, grid, cyclic, attrs

def calc_tend_sources(dat_mean, dat_inst, var, grid, cyclic, attrs, hor_avg=False, avg_dims=None):
    print("\nPrepare tendency calculations for {}".format(var.upper()))

    VAR = var.upper()
    dim_stag = None #for momentum: staggering dimension
    if var == "u":
        dim_stag = "x"
    elif var == "v":
        dim_stag = "y"
    elif var == "w":
        dim_stag = "bottom_top"

    #mapscale factors
    if var in ["u","v"]:
        mapfac_type = VAR
    else:
        mapfac_type = "M"
    mapfac_vnames = ["MAPFAC_{}X".format(mapfac_type),"MAPFAC_{}Y".format(mapfac_type)]
    mapfac = dat_inst[mapfac_vnames].isel(Time=0, drop=True)
    mapfac = mapfac.rename(dict(zip(mapfac_vnames,XY)))

    #map-scale factors for fluxes
    for d, m in zip(XY, ["UY", "VX"]):
        mf = dat_inst["MAPFAC_" + m].isel(Time=0, drop=True)
        flx = dat_mean["F{}{}_ADV_MEAN".format(var[0].upper(), d)]
        mapfac["F" + d] = stagger_like(mf, flx, cyclic=cyclic)

    dat_mean["FUY_SGS_MEAN"] = dat_mean["FVX_SGS_MEAN"]

    #density and dry air mass
    mu = grid["C2H"]+ grid["C1H"]*(dat_inst["MU"]+ dat_inst["MUB"])
    dat_inst["MU_STAG"] = mu
    grid["MU_STAG_MEAN"] = grid["C2H"]+ grid["C1H"]*dat_mean["MUT_MEAN"]
    rhodm = dat_mean["RHOD_MEAN"]
    if var in uvw:
        rhodm = stagger(rhodm, dim_stag, dat_inst[dim_stag + "_stag"], cyclic=cyclic[dim_stag], **grid[stagger_const])
        if var == "w":
            dat_inst["MU_STAG"] = grid["C2F"] + grid["C1F"]*(dat_inst["MU"]+ dat_inst["MUB"])
            grid["MU_STAG_MEAN"] = grid["C2F"] + grid["C1F"]*dat_mean["MUT_MEAN"]
        else:
            dat_inst["MU_STAG"] = stagger(dat_inst["MU_STAG"], dim_stag, dat_inst[dim_stag + "_stag"], cyclic=cyclic[dim_stag])
            grid["MU_STAG_MEAN"] = stagger(grid["MU_STAG_MEAN"], dim_stag, dat_mean[dim_stag + "_stag"], cyclic=cyclic[dim_stag])

    ref = rhodm
    if hor_avg:
        rhodm = avg_xy(rhodm, avg_dims, attrs)
    grid["RHOD_STAG_MEAN"] = rhodm

    #derivative of z wrt x,y,t
    dzdd = xr.Dataset()
    for d in xy:
        du = d.upper()
        dzdd[du] = diff(dat_mean["Z_MEAN"], d, dat_mean[d + "_stag"], cyclic=cyclic[d])/grid["D" + du]

    zw_inst = (dat_inst["PH"] + dat_inst["PHB"])/g
    dt = int(dat_inst.Time[1] - dat_inst.Time[0])*1e-9
    dzdd["T"] = zw_inst.diff("Time")/dt
    for d in [*XY, "T"]:
        dzdd[d] = stagger_like(dzdd[d], ref, ignore=["bottom_top_stag"], cyclic=cyclic)
    dzdd = remove_deprecated_dims(dzdd)
    grid["dzdd"] = dzdd.to_array("dir")

    for d, vel in zip(XY, ["u", "v"]):
        dph = -dat_mean["DPH_{}_MEAN".format(d)]/g
        grid["dzd{}_{}".format(d.lower(), vel)] = stagger_like(dph, ref, ignore=["bottom_top_stag"], cyclic=cyclic)

    rhod = - 1/diff(g*zw_inst, "bottom_top_stag", dat_inst.bottom_top)*grid["DNW"]*mu
    dat_inst["RHOD_STAG"] = stagger_like(rhod, ref, cyclic=cyclic, **grid[stagger_const])

    #height
    grid["ZW"] = dat_mean["Z_MEAN"]
    grid["Z_STAG"] = stagger_like(dat_mean["Z_MEAN"], ref, cyclic=cyclic, **grid[stagger_const])

    #additional sources
    print("Compute SGS and additional tendencies")

    sources = xr.Dataset()
    if var == "t":
        sources["mp"] = dat_mean["T_TEND_MP_MEAN"]
        sources["rad_lw"] = dat_mean["T_TEND_RADLW_MEAN"]
        sources["rad_sw"] = dat_mean["T_TEND_RADSW_MEAN"]
        if attrs["USE_THETA_M"] and (not attrs["OUTPUT_DRY_THETA_FLUXES"]):
            #convert sources from dry to moist theta
            sources = sources*(1 + rvovrd*dat_mean["Q_MEAN"])
            #add mp tendency
            sources["mp"] = sources["mp"] + dat_mean["Q_TEND_MP_MEAN"]*rvovrd*(dat_mean["T_MEAN"] + 300)
    elif var == "q":
        sources["mp"] = dat_mean["Q_TEND_MP_MEAN"]
    else:
        sources["pg"] = dat_mean["{}_TEND_PG_MEAN".format(VAR)]
        sources["cor_curv"] = dat_mean["{}_TEND_COR_CURV_MEAN".format(VAR)]

    #calculate tendencies from sgs fluxes and corrections
    sgs, sgsflux = sgs_tendency(dat_mean, VAR, grid, dzdd, cyclic, dim_stag=dim_stag, mapfac=mapfac)

    if hor_avg:
        sources = avg_xy(sources, avg_dims, attrs)
        sgs = avg_xy(sgs, avg_dims, attrs)
        sgsflux = avg_xy(sgsflux, avg_dims, attrs)
        grid = avg_xy(grid, avg_dims, attrs)

    sources = sources.to_array("comp")
    sources_sum = sources.sum("comp") + sgs.sum("dir", skipna=False)

    return dat_mean, dat_inst, sgs, sgsflux, sources, sources_sum, grid, dim_stag, mapfac


def build_mu(mut, grid, full_levels=False):
    if full_levels:
        mu = grid["C1F"]*mut + grid["C2F"]
    else:
        mu = grid["C1H"]*mut + grid["C2H"]
    return mu

def trb_fluxes(dat_mean, inst, variables, grid, cyclic, avg_kwargs, attrs, hor_avg=False, avg_dims=None):
    all_vars = ["RHOD_MEAN", "OMZN_MEAN"]
    for var in variables:
        for d,vel in zip(XYZ,uvw):
            all_vars.append(var.upper() + d + "_MEAN")
            all_vars.append(vel.upper() + "_MEAN")

    #fill all time steps with block average
    means = dat_mean[all_vars].reindex(Time=inst.Time).bfill("Time")
    if hor_avg:
        means = avg_xy(means, avg_dims, attrs, rho=means["RHOD_MEAN"], cyclic=cyclic,  **grid[stagger_const])
    for var in variables:
        var = var.upper()
        for d,vel in zip(["X","Y","Z", "Z"], ["U", "V", "W", "OMZN"]):
            var_d = var + d + "_MEAN"
            vel_m = vel + "_MEAN"
            #compute perturbations
            var_pert = inst[var_d] - means[var_d]
            vel_pert = stagger_like(inst[vel_m] - means[vel_m], var_pert, cyclic=cyclic, **grid[stagger_const])
            rho_stag = stagger_like(inst["RHOD_MEAN"], var_pert, cyclic=cyclic, **grid[stagger_const])
            rho_stag_mean = stagger_like(dat_mean["RHOD_MEAN"], var_pert, cyclic=cyclic, **grid[stagger_const])
            flux = rho_stag*vel_pert*var_pert
            flux = flux.coarsen(**avg_kwargs).mean()/rho_stag_mean
            if hor_avg:
                flux = avg_xy(flux, avg_dims, attrs)
            dat_mean["F{}{}_TRB_MEAN".format(var, vel)] = flux