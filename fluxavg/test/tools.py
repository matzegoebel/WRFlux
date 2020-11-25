#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 12:53:04 2019

@author: csat8800
"""
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
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
tex_names = {"t" : "\\theta", "q" : "q_\\mathrm{v}"}
units_dict = {"t" : "K ", "q" : "", **{v : "ms$^{-1}$" for v in uvw}}
units_dict_tend = {"t" : "Ks$^{-1}$", "q" : "s$^{-1}$", **{v : "ms$^{-2}$" for v in uvw}}
units_dict_flux = {"t" : "Kms$^{-1}$", "q" : "ms$^{-1}$", **{v : "m$^{2}$s$^{-2}$" for v in uvw}}
units_dict_tend_rho = {"t" : "kg m$^{-3}$Ks$^{-1}$", "q" : "kg m$^{-3}$s$^{-1}$", **{v : "kg m$^{-2}$s$^{-2}$" for v in uvw}}
g = 9.81
rvovrd = 461.6/287.04

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


def avg_xy(data, avg_dims):
    """Average data over the given dimensions even
    if the actual present dimension has '_stag' added"""

    avg_dims_final = avg_dims.copy()
    for i,d in enumerate(avg_dims):
        if (d + "_stag" in data.dims):
            avg_dims_final.append(d + "_stag")
            if (d not in data.dims):
                avg_dims_final.remove(d)

    return data.mean(avg_dims_final)

def find_nans(dat):
    """Drop all indeces of each dimension that do not contain NaNs"""
    nans = dat.isnull()
    nans = nans.where(nans)
    nans = dropna_dims(nans)
    dat = dat.loc[nans.indexes]

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
    data : xarray dataarray
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
    data : xarray dataarray
        output data.

    """
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


#%% WRF tendencies

def sgs_tendency(dat_mean, VAR, grid, dzdd, cyclic, dim_stag=None, mapfac=None, **stagger_const):
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
    rhoz = stagger_like(dat_mean["RHOD_MEAN"], fz, cyclic=cyclic, **stagger_const)
    sgs["Z"] = -diff(fz*rhoz, d3s, new_coord=vcoord)
    sgs["Z"] = sgs["Z"]/dn/grid["MU_STAG"]*(-g)
    for d, v in zip(xy, ["U", "V"]):
        #compute corrections
        du = d.upper()
        if mapfac is None:
            m = 1
        else:
            m = mapfac[du]
        fd = dat_mean["F{}{}_SGS_MEAN".format(VAR, du)]
        sgsflux[du] = fd
        fd = fd*stagger_like(dat_mean["RHOD_MEAN"], fd, cyclic=cyclic, **stagger_const)
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
            flux8z = stagger(flux8v, d3, grid["ZNW"], **stagger_const)
            flux8z[:,[0,-1]] = 0
        corr_sgs = diff(flux8z, d3s, new_coord=vcoord)/dn
        corr_sgs = corr_sgs*stagger_like(dzdd[du], corr_sgs, cyclic=cyclic)

        dx = grid["D" + du]
        sgs[du] = -diff(fd, ds, new_coord=sgs[d], cyclic=cyc)/dx*m
        if VAR == "W":
            m = mapfac["Y"]
        sgs[du] = sgs[du]/dat_mean["RHOD_MEAN_STAG"] + corr_sgs*m/grid["MU_STAG"]*(-g)

    sgsflux["Z"] = fz
    sgs = sgs[XYZ]
    sgs = sgs.to_array("dir")
    if VAR == "W":
        sgs[:,:,[0,-1]] = 0

    return sgs, sgsflux


def adv_tend(dat_mean, VAR, var_stag, grid, mapfac, cyclic, stagger_const, cartesian=False,
             hor_avg=False, avg_dims=None, fluxnames=None, w=None):

    print("Comute resolved tendencies")

    if fluxnames is None:
        fluxnames = ["F{}{}_ADV_MEAN".format(VAR, d) for d in XYZ]

    tot_flux = dat_mean[fluxnames]
    tot_flux = tot_flux.rename(dict(zip(fluxnames, XYZ)))
    rhod8z = stagger_like(dat_mean["RHOD_MEAN"], tot_flux["Z"], cyclic=cyclic, **stagger_const)
    if not cartesian:
          tot_flux["Z"] = tot_flux["Z"] - (dat_mean["F{}X_CORR".format(VAR)] + \
              dat_mean["F{}Y_CORR".format(VAR)] + dat_mean["CORR_D{}DT".format(VAR)])/rhod8z

    #get vertical velocity
    if w is None:
        if cartesian:
            w = dat_mean["WD_MEAN"]
        else:
            rhod = stagger(dat_mean["RHOD_MEAN"], "bottom_top", dat_mean["bottom_top_stag"], **stagger_const)
            w = dat_mean["WW_MEAN"]/(-g*rhod)

    vmean = xr.Dataset({"X" : dat_mean["U_MEAN"], "Y" : dat_mean["V_MEAN"], "Z" : w})
    if hor_avg:
        for k in vmean.keys():
            vmean[k] = avg_xy(vmean[k], avg_dims)

  #  mean advective fluxes
    mean_flux = xr.Dataset()
    for d in XYZ:
        if hor_avg and (d.lower() in avg_dims):
            mean_flux[d] = 0.
        else:
            vel_stag = stagger_like(vmean[d], ref=var_stag[d], cyclic=cyclic, **stagger_const)
            var_stag_d = var_stag[d]
            mean_flux[d] = var_stag_d*vel_stag

    #advective tendency from fluxes
    adv = {}
    fluxes = {"adv_r" : tot_flux, "mean" : mean_flux}
    for comp, flux in fluxes.items():
        adv_i = xr.Dataset()
        mf = mapfac
        rhod8z_m = rhod8z
        if (comp == "mean") and hor_avg:
            mf = avg_xy(mapfac, avg_dims)
            rhod8z_m = avg_xy(rhod8z, avg_dims)
        for d in xy:
            du = d.upper()
            cyc = cyclic[d]
            if hor_avg and (d in avg_dims) and (comp == "mean"):
                adv_i[du] = 0.
                continue
            if d in flux[du].dims:
                ds = d
                d = d + "_stag"
            else:
                ds = d + "_stag"
            dx = grid["D" + du]

            mf_flx = mapfac["F" + du]
            mu = dat_mean["MUT_MEAN"]
            if (comp == "mean") and hor_avg:
                mf_flx = avg_xy(mf_flx, avg_dims)
                mu = avg_xy(mu, avg_dims)
            mu = build_mu(mu, flux[du], grid, cyclic=cyclic)
            adv_i[du] = -diff(mu*flux[du]/mf_flx, ds, dat_mean[d], cyclic=cyc)*mf["X"]*mf["Y"]/dx
        fz = rhod8z_m*flux["Z"]
        if VAR == "W":
            adv_i["Z"] = -diff(fz, "bottom_top", grid["ZNW"])/grid["DN"]
            #set sfc and top point correctly
            adv_i["Z"][{"bottom_top_stag" : 0}] = 0.
            adv_i["Z"][{"bottom_top_stag" : -1}] = (2*fz.isel(bottom_top=-1)/grid["DN"][-2]).values

        else:
            adv_i["Z"] = -diff(fz, "bottom_top_stag", grid["ZNU"])/grid["DNW"]
        adv_i["Z"] = adv_i["Z"]*(-g)
        adv_i = adv_i/grid["MU_STAG"]
        adv[comp] = adv_i

    if hor_avg:
        adv["adv_r"] = avg_xy(adv["adv_r"], avg_dims)
        fluxes["adv_r"] = avg_xy(fluxes["adv_r"], avg_dims)

    keys = adv.keys()
    adv = xr.concat(adv.values(), "comp")
    adv = adv.to_array("dir")
    adv["comp"] = list(keys)
    flux = xr.concat(fluxes.values(), "comp")
    flux["comp"] = list(fluxes.keys())

    #resolved turbulent fluxes and tendencies
    flux = flux.reindex(comp=["adv_r", "mean", "trb_r"])
    adv = adv.reindex(comp=["adv_r", "mean", "trb_r"])
    for d in flux.data_vars:
        flux[d].loc["trb_r"] = flux[d].loc["adv_r"] - flux[d].loc["mean"]
        adv.loc[d, "trb_r"] = adv.loc[d, "adv_r"] - adv.loc[d, "mean"]

    return flux, adv, vmean

def cartesian_corrections(VAR, dim_stag, corr, var_stag, vmean, rhodm, dzdd, grid, mapfac, adv, tend,
                          cyclic, stagger_const, hor_avg=False, avg_dims=None):

    print("Compute Cartesian corrections")
    #decompose cartesian corrections
    #total
    corr = corr.to_array("dim").expand_dims(comp=["adv_r"]).reindex(comp=["adv_r", "mean", "trb_r"])
    if hor_avg:
        corr = avg_xy(corr, avg_dims)
        rhodm = avg_xy(rhodm, avg_dims)
        dzdd = avg_xy(dzdd, avg_dims)
    #mean part
    for i, (d, v) in enumerate(zip(xy, ["U", "V"])):
        #staggering
        if hor_avg and (d in avg_dims):
            corr.loc["mean"][i] = 0
            continue
        du = d.upper()
        kw = dict(ref=var_stag["Z"], cyclic=cyclic, **stagger_const)
        vmean_stag =  stagger_like(vmean[d.upper()], **kw)
        rho_stag =  stagger_like(rhodm, **kw)

        corr.loc["mean"][i] = rho_stag*vmean_stag*var_stag["Z"]*stagger_like(dzdd[du], **kw)

    #resolved turbulent part
    corr.loc["trb_r"] = corr.loc["adv_r"] - corr.loc["mean"]

    #correction flux to tendency
    if "W" in VAR:
        dcorr_dz = diff(corr, "bottom_top", grid["ZNW"])/grid["DN"]
        dcorr_dz[{"bottom_top_stag" : 0}] = 0.
        dcorr_dz[{"bottom_top_stag" : -1}] = -(2*corr.isel(bottom_top=-1)/grid["DN"][-2]).values
    else:
        dcorr_dz = diff(corr, "bottom_top_stag", grid["ZNU"])/grid["DNW"]
    dcorr_dz = dcorr_dz/grid["MU_STAG"]*(-g)
    #apply corrections
    for i, d in enumerate(XY):
        adv.loc[d] = adv.loc[d] + dcorr_dz[:, i]
    tend = tend - dcorr_dz.sel(comp="adv_r", drop=True)[2]

    return adv, tend, corr

def total_tendency(dat_inst, var, **attrs):
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
    rvar = vard*dat_inst["MU_STAG"]

    # total tendency
    dt = int(dat_inst.Time[1] - dat_inst.Time[0])*1e-9
    total_tend = rvar.diff("Time")/dt

    return total_tend
#%%prepare variables
def prepare(dat_mean, dat_inst, t_avg=False, t_avg_interval=None):
    print("Prepare data")
    attrs = dat_inst.attrs
    dat_inst.attrs = {}

    #strip first time as it contains only zeros
    dat_mean = dat_mean.sel(Time=dat_mean.Time[1:])

    if t_avg:
        avg_kwargs = dict(Time=t_avg_interval, coord_func={"Time" : partial(select_ind, indeces=-1)}, boundary="trim")
        dat_mean = dat_mean.coarsen(**avg_kwargs).mean()

    #computational grid
    grid = dat_inst[["ZNU","ZNW","DNW","DN","C1H","C2H","C1F","C2F"]].isel(Time=0, drop=True)
    grid["DN"] = grid["DN"].rename(bottom_top="bottom_top_stag").assign_coords(bottom_top_stag=grid["ZNW"][:-1]).reindex(bottom_top_stag=grid["ZNW"])
    grid["DX"] = attrs["DX"]
    grid["DY"] = attrs["DY"]
    stagger_const = dat_inst[["FNP", "FNM", "CF1", "CF2", "CF3", "CFN", "CFN1"]].isel(Time=0, drop=True)

    dat_mean = dat_mean.assign_coords(bottom_top=grid["ZNU"], bottom_top_stag=grid["ZNW"])
    dat_inst = dat_inst.assign_coords(bottom_top=grid["ZNU"], bottom_top_stag=grid["ZNW"])
    #select start and end points of averaging intervals
    dat_inst = dat_inst.sel(Time=[dat_inst.Time[0].values, *dat_mean.Time.values])
    for v in dat_inst.coords:
        if ("XLAT" in v) or ("XLONG" in v):
            dat_inst = dat_inst.drop(v)

    #check if periodic bc can be used in staggering operations
    cyclic = {d : bool(attrs["PERIODIC_{}".format(d.upper())]) for d in xy}
    cyclic["bottom_top"] = False

    return dat_mean, dat_inst, grid, cyclic, stagger_const, attrs

def calc_tend_sources(dat_mean, dat_inst, var, grid, cyclic, stagger_const, attrs, hor_avg=False, avg_dims=None):
    print("\nPrepare tendency calculations for {}".format(var.upper()))
    dt = int(dat_mean.Time[1] - dat_mean.Time[0])*1e-9

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
    rhodm = dat_mean["RHOD_MEAN"]
    rhodm8z = stagger(rhodm, "bottom_top", grid["ZNW"], **stagger_const)
    dat_inst["MU_STAG"] = grid["C2H"]+ grid["C1H"]*(dat_inst["MU"]+ dat_inst["MUB"])
    grid["MU_STAG"] = grid["C2H"]+ grid["C1H"]*dat_mean["MUT_MEAN"]
    if var in uvw:
        rhodm = stagger(rhodm, dim_stag, dat_inst[dim_stag + "_stag"], cyclic=cyclic[dim_stag], **stagger_const)
        if var == "w":
            dat_inst["MU_STAG"] = grid["C2F"] + grid["C1F"]*(dat_inst["MU"]+ dat_inst["MUB"])
            grid["MU_STAG"] = grid["C2F"] + grid["C1F"]*dat_mean["MUT_MEAN"]
        else:
            dat_inst["MU_STAG"] = stagger(dat_inst["MU_STAG"], dim_stag, dat_inst[dim_stag + "_stag"], cyclic=cyclic[dim_stag])
            grid["MU_STAG"] = stagger(grid["MU_STAG"], dim_stag, dat_mean[dim_stag + "_stag"], cyclic=cyclic[dim_stag])
    if hor_avg:
        rhodm = avg_xy(rhodm, avg_dims)
    dat_mean["RHOD_MEAN_STAG"] = rhodm

    #calculate total tendency
    total_tend = total_tendency(dat_inst, var, **attrs)/grid["MU_STAG"]

    #derivative of z wrt x,y,t
    dzdd = xr.Dataset()
    for d in xy:
        du = d.upper()
        dzdd[du] = diff(dat_mean["Z_MEAN"], d, dat_mean[d + "_stag"], cyclic=cyclic[d])/grid["D" + du]

    zw_inst = (dat_inst["PH"] + dat_inst["PHB"])/g
    dzdd["T"] = zw_inst.diff("Time")/dt
    for d in [*XY, "T"]:
        dzdd[d] = stagger_like(dzdd[d], total_tend, ignore=["bottom_top_stag"], cyclic=cyclic)

    #vertically destagger for w
    dzdd_s = dzdd.copy()
    if var == "w":
        dzdd_s = destagger(dzdd_s, "bottom_top_stag", grid["ZNU"])

    #diagnostic vertically velocity, correctly staggered
    dat_mean["WD_MEAN"] = dzdd_s["T"] +  stagger_like(dat_mean["WW_MEAN"]/(-g*rhodm8z), dzdd_s["T"], cyclic=cyclic, **stagger_const)

    for d,v in zip(XY, ["U","V"]):
        dat_mean["WD_MEAN"] = dat_mean["WD_MEAN"] + dzdd_s[d]*stagger_like(dat_mean[v + "_MEAN"], dzdd_s[d], cyclic=cyclic, **stagger_const)

    #height
    grid["ZW"] = dat_mean["Z_MEAN"]
    grid["Z_STAG"] = stagger_like(dat_mean["Z_MEAN"], total_tend, cyclic=cyclic)

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
            sources = sources + dat_mean["Q_TEND_MP_MEAN"]*rvovrd*(dat_mean["T_MEAN"] + 300)
    elif var == "q":
        sources["mp"] = dat_mean["Q_TEND_MP_MEAN"]
    else:
        sources["pg"] = dat_mean["{}_TEND_PG_MEAN".format(VAR)]
        sources["cor_curv"] = dat_mean["{}_TEND_COR_CURV_MEAN".format(VAR)]

    #calculate tendencies from sgs fluxes and corrections
    sgs, sgsflux = sgs_tendency(dat_mean, VAR, grid, dzdd, cyclic, dim_stag=dim_stag, mapfac=mapfac, **stagger_const)

    var_stag = xr.Dataset()
    #get variable staggered in flux direction
    for d in XYZ:
        var_stag[d] = dat_mean["{}{}_MEAN".format(VAR, d)]

    if hor_avg:
        sources = avg_xy(sources, avg_dims)
        sgs = avg_xy(sgs, avg_dims)
        sgsflux = avg_xy(sgsflux, avg_dims)
        total_tend = avg_xy(total_tend, avg_dims)
        var_stag = avg_xy(var_stag, avg_dims)
        grid = avg_xy(grid, avg_dims)

    sources = sources.to_array("comp")
    sources_sum = sources.sum("comp") + sgs.sum("dir", skipna=False)

    return dat_mean, dat_inst, total_tend, sgs, sgsflux, sources, sources_sum, var_stag, grid, dim_stag, mapfac, dzdd, dzdd_s


def build_mu(mut, ref, grid, cyclic=None):
    mu = stagger_like(mut, ref, cyclic=cyclic)
    if "bottom_top" in ref.dims:
        mu = grid["C1H"]*mu + grid["C2H"]
    else:
        mu = grid["C1F"]*mu + grid["C2F"]
    return mu
#%% plotting

def scatter_tend_forcing(tend, forcing, var, plot_diff=False, hue="eta", savefig=True, title=None, fname=None, figloc=figloc, **kwargs):
    print("scatter plot")
    pdat = xr.concat([tend, forcing], "comp")

    if plot_diff:
        pdat[1] = pdat[1] - pdat[0]
    pdatf = pdat[0].stack(s=pdat[0].dims)

    #set color
    if hue == "eta":
        if "bottom_top" in dir(pdatf):
            color = -pdatf.bottom_top
        else:
            color = -pdatf.bottom_top_stag
    elif hue == "Time":
        color = pdatf.Time
    elif hue == "x":
        color = pdatf.x
    elif hue == "y":
        color = pdatf.y
    else:
        raise ValueError("Hue {} not supported".format(hue))

    if hue != "eta":
        color = (color - color.min())/(color.max() - color.min())

    fig, ax = plt.subplots()
    kwargs.setdefault("s", 10)
    kwargs.setdefault("cmap", "cool")
    p = plt.scatter(pdat[0], pdat[1], c=color.values, **kwargs)

    for i in [0,1]:
        pdat = pdat.where(~pdat[i].isnull())
    if not plot_diff:
        minmax = [pdat.min(), pdat.max()]
        dist = minmax[1] - minmax[0]
        minmax[0] -= 0.03*dist
        minmax[1] += 0.03*dist
        plt.plot(minmax, minmax, c="k")
        ax.set_xlim(minmax)
        ax.set_ylim(minmax)

    if var in tex_names:
        tex_name = tex_names[var]
    else:
        tex_name = var
    xlabel = "Total ${}$ tendency".format(tex_name)
    ylabel = "Total ${}$ forcing".format(tex_name)
    if plot_diff:
        ylabel += " - " + xlabel
    units = " ({})".format(units_dict_tend[var])
    plt.xlabel(xlabel + units)
    plt.ylabel(ylabel + units)

    #colorbar
    cax = fig.add_axes([0.84,0.1,0.1,0.8], frameon=False)
    cax.set_yticks([])
    cax.set_xticks([])
    if hue == "eta":
        cb = plt.colorbar(p,ax=cax,label="$\eta$")
        cb.set_ticks(np.arange(-0.8,-0.2,0.2))
        cb.set_ticklabels(np.linspace(0.8,0.2,4).round(1))
    else:
        plt.colorbar(p,ax=cax,label="hour")

    #error labels
    err = abs(tend - forcing)
    rmse = (err**2).mean().values**0.5
    r2 = np.corrcoef(tend.values.flatten(), forcing.values.flatten())[1,0]
    ax.text(0.74,0.07,"RMSE={0:.2E}\nR$^2$={1:.3f}".format(rmse, r2), horizontalalignment='left',
             verticalalignment='bottom', transform=ax.transAxes)


    if title is None:
        title = fname
    fig.suptitle(title)

    if savefig:
        fig.savefig(figloc + "{}_budget/scatter/{}.png".format(var, fname),dpi=300, bbox_inches="tight")

    return fig