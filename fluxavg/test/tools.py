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
uvw = ["u", "v", "w"]
tex_names = {"t" : "\\theta", "q" : "q_\\mathrm{v}"}
units_dict = {"t" : "K ", "q" : "", **{v : "ms$^{-1}$" for v in uvw}}
units_dict_tend = {"t" : "Ks$^{-1}$", "q" : "s$^{-1}$", **{v : "ms$^{-2}$" for v in uvw}}
units_dict_tend_rho = {"t" : "kg m$^{-3}$Ks$^{-1}$", "q" : "kg m$^{-3}$s$^{-1}$", **{v : "kg m$^{-2}$s$^{-2}$" for v in uvw}}
g = 9.81
rvovrd = 461.6/287.04
#%%definitions
host = socket.gethostname()
if host == 'pc45-c707':
    basedir = "/media/HDD/Matthias/phd/"
elif host in ["matze-thinkpad", "pc26-c707"]:
    basedir = "~/phd/"
else:
    basedir = "/scratch/c7071088/phd/"

basedir = os.path.expanduser(basedir)

figloc = basedir + "figures/"

#%%open dataset
def fix_coords(data, dx, dy):
    """Assign time and space coordinates"""

    #assign time coordinate
    if "XTIME" in data:
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

def open_dataset(file, var=None, chunks=None, del_attrs=True,fix_c=True,ht=None, **kwargs):

    ds = xr.open_dataset(file, **kwargs)
    if fix_c:
        dx, dy = ds.DX, ds.DY
        ds = fix_coords(ds, dx=dx, dy=dy)

    if var is not None:
        var = make_list(var)
        ds = ds[var]

    if chunks is not None:
        ds = chunk_data(ds, chunks)

    if (ht is not None) and ("bottom_top" in ds.dims):
        ds = ds.assign_coords(height=ht)
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

def sgs_tendency(dat, VAR, grid, dzdd, cyclic, dim_stag=None, mapfac=None, **stagger_const):
    sgs = xr.Dataset()
    if VAR == "W":
        d3s = "bottom_top"
        d3 = "bottom_top_stag"
        vcoord = grid["ZNW"]
        dz = grid["DZ"]
    else:
        d3 = "bottom_top"
        d3s = "bottom_top_stag"
        vcoord = grid["ZNU"]
        dz = grid["DZW"]

    sgs["Z"] = -diff(dat["F{}Z_SGS_MEAN".format(VAR)], d3s, new_coord=vcoord)
    dz_stag = stagger_like(dz, sgs["Z"], cyclic=cyclic, **stagger_const)
    sgs["Z"] = sgs["Z"]/dz_stag
    for d, v in zip(["x", "y"], ["U", "V"]):
        #compute corrections
        du = d.upper()
        if mapfac is None:
            m = 1
        else:
            m = mapfac[du]
        sgsflux = dat["F{}{}_SGS_MEAN".format(VAR, du)]
        cyc = cyclic[d]
        if d in sgsflux.dims:
            #for momentum variances
            ds = d
            d = ds + "_stag"
            flux8v = stagger(sgsflux, ds, new_coord=sgs[d], cyclic=cyc)
        else:
            ds = d + "_stag"
            flux8v = destagger(sgsflux, ds, new_coord=sgs[d])

        if VAR == "W":
            flux8z = destagger(flux8v, d3, grid["ZNU"])
        else:
            flux8z = stagger(flux8v, d3, grid["ZNW"], **stagger_const)
            flux8z[:,[0,-1]] = 0
        corr_sgs = diff(flux8z, d3s, new_coord=vcoord)/dz_stag
        corr_sgs = corr_sgs*stagger_like(dzdd[du], corr_sgs, cyclic=cyclic)

        dx = grid["D" + du]
        sgs[du] = -diff(sgsflux, ds, new_coord=sgs[d], cyclic=cyc)/dx*m
        if VAR == "W":
            m = mapfac["Y"]
        sgs[du] = sgs[du] + corr_sgs*m
    sgs = sgs.to_array("dir")
    if VAR == "W":
        sgs[:,:,[0,-1]] = 0

    return sgs


def adv_tend(data, VAR, var_stag, grid, mapfac, cyclic, stagger_const, cartesian=False,
             recalc_w=True, hor_avg=False, avg_dims=None, fluxnames=None, w=None):


    if fluxnames is None:
        fluxnames = ["F{}{}_ADV_MEAN".format(VAR, d) for d in ["X", "Y", "Z"]]

    #get vertical velocity
    if w is None:
        if cartesian:
            w = data["WD_MEAN"]
        else:
            w = data["WW_MEAN"]/(-g)
    vmean = {"X" : data["U_MEAN"], "Y" : data["V_MEAN"], "Z" : w}
    if hor_avg:
        for k in vmean.keys():
            vmean[k] = avg_xy(vmean[k], avg_dims)
    tot_flux = data[fluxnames]
    tot_flux = tot_flux.rename(dict(zip(fluxnames, ["X", "Y", "Z"])))
    tot_flux["Z"] = tot_flux["Z"]*stagger_like(data["RHOD_MEAN"], tot_flux["Z"], cyclic=cyclic, **stagger_const)
    if not cartesian:
          tot_flux["Z"] = tot_flux["Z"] - data["F{}X_CORR".format(VAR)] - \
              data["F{}Y_CORR".format(VAR)] - data["CORR_D{}DT".format(VAR)]

  #  mean advective fluxes
    mean_flux = xr.Dataset()
    for d in ["X", "Y", "Z"]:
        if hor_avg and (d.lower() in avg_dims):
            continue
        kw = dict(ref=var_stag[d], cyclic=cyclic, **stagger_const)
        rho_stag = stagger_like(data["RHOD_MEAN"], **kw).drop("z")
        vel_stag = stagger_like(vmean[d], **kw)
        if (VAR != "W") or cartesian:
            vel_stag = rho_stag*vel_stag
        var_stag_d = var_stag[d]
        mean_flux[d] = var_stag_d*vel_stag

    #advective tendency from fluxes
    adv = {}
    fluxes = {"tot" : tot_flux, "mean" : mean_flux}
    for comp, flux in fluxes.items():
        adv_i = xr.Dataset()
        mf = mapfac
        if (comp == "mean") and hor_avg:
            mf = avg_xy(mapfac, avg_dims)

        for d in ["x", "y"]:
            du = d.upper()
            cyc = cyclic[d]
            if du not in flux.data_vars:
                continue
            if d in flux[du].dims:
                ds = d
                d = d + "_stag"
            else:
                ds = d + "_stag"
            dx = grid["D" + du]
            mu = build_mu(data["MUT_MEAN"], flux[du], grid, cyclic=cyclic)
            adv_i[du] = -diff(mu*flux[du], ds, data[d], cyclic=cyc)*mf["X"]*mf["Y"]/dx

        #TODO mult with rhod
        if VAR == "W":
            adv_i["Z"] = -diff(flux["Z"], "bottom_top", grid["ZNW"])/grid["DN"]
            #TODO: not quite correct for top
            adv_i = adv_i.where((adv_i.bottom_top_stag > 0) * (adv_i.bottom_top_stag < 1) , 0)
        else:
            adv_i["Z"] = -diff(flux["Z"], "bottom_top_stag", grid["ZNU"])/grid["DNW"]
        if not cartesian:#TODO still correct?
            adv_i["Z"] = adv_i["Z"]*mf["Y"]
        adv_i["Z"] = adv_i["Z"]*(-g)
        adv_i = adv_i/data["MU_STAG"]
        adv[comp] = adv_i

    if hor_avg:
        adv["tot"] = avg_xy(adv["tot"], avg_dims)
        fluxes["tot"] = avg_xy(fluxes["tot"], avg_dims)
        for d in avg_dims:
            #put total advective component of averaged dimension in mean flux (not yet set)
            adv["mean"][d.upper()] = adv["tot"][d.upper()]
            fluxes["mean"][d.upper()] = fluxes["tot"][d.upper()]
    keys = adv.keys()
    adv = xr.concat(adv.values(), "comp")
    adv = adv.to_array("dir")
    adv["comp"] = list(keys)
    flux = xr.concat(fluxes.values(), "comp")
    flux["comp"] = list(fluxes.keys())

    #resolved turbulent fluxes and tendencies
    flux = flux.reindex(comp=["tot", "mean", "res"])
    adv = adv.reindex(comp=["tot", "mean", "res"])
    for d in flux.data_vars:
        flux[d].loc["res"] = flux[d].loc["tot"] - flux[d].loc["mean"]
        adv.loc[d, "res"] = adv.loc[d, "tot"] - adv.loc[d, "mean"]

    return flux, adv, vmean

def cartesian_corrections(VAR, dim_stag, corr, var_stag, vmean, rhodm, dzdd, grid, adv, tend, cyclic, stagger_const):
    #decompose cartesian corrections
    #total
    corr = corr.expand_dims(comp=["tot"]).reindex(comp=["tot", "mean", "res"])
    #mean part
    for i, (d, v) in enumerate(zip(["x", "y"], ["U", "V"])):
        #staggering
        du = d.upper()
        kw = dict(ref=var_stag["Z"], cyclic=cyclic, **stagger_const)
        vmean_stag =  stagger_like(vmean[d.upper()], **kw)
        rho_stag =  stagger_like(rhodm, **kw)

        corr.loc["mean"][i] = rho_stag*vmean_stag*var_stag["Z"]*stagger_like(dzdd[du], **kw)

    #resolved turbulent part
    corr.loc["res"] = corr.loc["tot"] - corr.loc["mean"]

    #correction flux to tendency
    if VAR == "W":
        dcorr_dz = diff(corr, "bottom_top", grid["ZNW"])
        dz_stag = grid["DZ"]
    else:
        dcorr_dz = diff(corr, "bottom_top_stag", grid["ZNU"])
        dz_stag = grid["DZW"]
        if VAR in ["U", "V"]:
            dz_stag = stagger(grid["DZW"], dim_stag, dcorr_dz[dim_stag + "_stag"], cyclic=cyclic[dim_stag], **stagger_const)
    dcorr_dz = dcorr_dz/dz_stag

    #apply corrections
    for i, d in enumerate(["X", "Y"]):
        adv.loc[d] = adv.loc[d] + dcorr_dz[:, i]
    tend = tend - dcorr_dz.sel(comp="tot", drop=True)[2]

    return adv, tend

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
    grid["ZW"] = dat_mean["Z_MEAN"]
    grid["Z"] = destagger(dat_mean["Z_MEAN"], "bottom_top_stag", grid["ZNU"])
    grid["DZW"] = diff(grid["ZW"], "bottom_top_stag", grid["ZNU"])
    grid["DZ"] = diff(grid["Z"], "bottom_top", grid["ZNW"])
    grid["DZDNW"] = grid["DZW"]/grid["DNW"]
    grid["DZDN"] = grid["DZ"]/grid["DN"]

    stagger_const = dat_inst[["FNP", "FNM", "CF1", "CF2", "CF3", "CFN", "CFN1"]].isel(Time=0, drop=True)

    dat_mean = dat_mean.assign_coords(bottom_top=grid["ZNU"], bottom_top_stag=grid["ZNW"], z=grid["Z"], zw=grid["ZW"])
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
    mapfac = mapfac.rename(dict(zip(mapfac_vnames,["X","Y"])))

    #adjust fluxes
    # for f in ["RU", "RV", "WW"]:
        #divide total fluxes by gravity
   # f = "WW" + VAR + "_ADV_MEAN"
   # dat_mean[f] = dat_mean[f]/(-9.81)
    dat_mean["FUY_SGS_MEAN"] = dat_mean["FVX_SGS_MEAN"]
    #resolved vertical flux with diagnosed vertical velocity

    #density and dry air mass
    rhodm = dat_mean["RHOD_MEAN"]
    rhodm8z = stagger(rhodm, "bottom_top", grid["ZNW"], **stagger_const)
    dat_inst["MU_STAG"] = grid["C2H"]+ grid["C1H"]*(dat_inst["MU"]+ dat_inst["MUB"])
    dat_mean["MU_STAG"] = grid["C2H"]+ grid["C1H"]*dat_mean["MUT_MEAN"]
    if var in uvw:
        dat_mean["RHOD_MEAN_STAG"] = stagger(rhodm, dim_stag, dat_inst[dim_stag + "_stag"], cyclic=cyclic[dim_stag], **stagger_const)
        if var == "w":
            dat_inst["MU_STAG"] = grid["C2F"] + grid["C1F"]*(dat_inst["MU"]+ dat_inst["MUB"])
            dat_mean["MU_STAG"] = grid["C2F"] + grid["C1F"]*dat_mean["MUT_MEAN"]
        else:
            dat_inst["MU_STAG"] = stagger(dat_inst["MU_STAG"], dim_stag, dat_inst[dim_stag + "_stag"], cyclic=cyclic[dim_stag])
            dat_mean["MU_STAG"] = stagger(dat_mean["MU_STAG"], dim_stag, dat_mean[dim_stag + "_stag"], cyclic=cyclic[dim_stag])
    else:
        dat_mean["RHOD_MEAN_STAG"] = rhodm

    #calculate total tendency
    total_tend = total_tendency(dat_inst, var, **attrs)/dat_mean["MU_STAG"]

    if var in uvw:
        if var == "w":
            grid["DZDN_STAG"] = grid["DZDN"]
        else:
            grid["DZDN_STAG"] = stagger_like(grid["DZDNW"], total_tend, cyclic=cyclic, **stagger_const)
    else:
        grid["DZDN_STAG"] = grid["DZDNW"]

    #derivative of z wrt x,y,t
    dzdd = xr.Dataset()
    for d in xy:
        du = d.upper()
        dzdd[du] = diff(grid["ZW"], d, dat_mean[d + "_stag"], cyclic=cyclic[d])/grid["D" + du]

    dzdd["T"] = grid["ZW"].diff("Time")/dt
    for d in [*XY, "T"]:
        dzdd[d] = stagger_like(dzdd[d], total_tend, ignore=["bottom_top_stag"], cyclic=cyclic)

    #vertically destagger for w
    dzdd_s = dzdd.copy()
    if var == "w":
        dzdd_s = destagger(dzdd_s, "bottom_top_stag", grid["ZNU"])

    #diagnostic vertically velocity, correctly staggered
    dat_mean["WD_MEAN"] = dzdd_s["T"] +  stagger_like(dat_mean["WW_MEAN"]/(-g*rhodm8z), dzdd_s["T"], cyclic=cyclic, **stagger_const)

    for d,v in zip(["X","Y"], ["U","V"]):
        dat_mean["WD_MEAN"] = dat_mean["WD_MEAN"] + dzdd_s[d]*stagger_like(dat_mean[v + "_MEAN"], dzdd_s[d], cyclic=cyclic, **stagger_const)

    #additional sources
    if var == "t":
        sources = dat_mean["T_TEND_MP_MEAN"] + dat_mean["T_TEND_RADSW_MEAN"] + dat_mean["T_TEND_RADLW_MEAN"]
        if attrs["USE_THETA_M"] and (not attrs["OUTPUT_DRY_THETA_FLUXES"]):
            #convert sources from dry to moist theta
            sources = sources*(1 + rvovrd*dat_mean["Q_MEAN"])
            #add mp tendency
            sources = sources + dat_mean["Q_TEND_MP_MEAN"]*rvovrd*(dat_mean["T_MEAN"] + 300)
    elif var == "q":
        sources = dat_mean["Q_TEND_MP_MEAN"]
    else:
        sources = dat_mean["{}_TEND_PG_MEAN".format(VAR)] + dat_mean["{}_TEND_COR_CURV_MEAN".format(VAR)]

    #calculate tendencies from sgs fluxes and corrections
    sgs = sgs_tendency(dat_mean, VAR, grid, dzdd, cyclic, dim_stag=dim_stag, mapfac=mapfac, **stagger_const)

    sources = sources + sgs.sum("dir", skipna=False)/dat_mean["RHOD_MEAN_STAG"]

    if hor_avg:
        sources = avg_xy(sources, avg_dims)
        total_tend = avg_xy(total_tend, avg_dims)
        dat_mean["RHOD_MEAN"] = avg_xy(dat_mean["RHOD_MEAN"], avg_dims)
        dat_mean["RHOD_MEAN_STAG"] = avg_xy(dat_mean["RHOD_MEAN_STAG"], avg_dims)
        grid = avg_xy(grid, avg_dims)

    return dat_mean, dat_inst, total_tend, sgs, sources, grid, dim_stag, mapfac, dzdd, dzdd_s


def build_mu(mut, ref, grid, cyclic=None):
    mu = stagger_like(mut, ref, cyclic=cyclic)
    if "bottom_top" in ref.dims:
        mu = grid["C1H"]*mu + grid["C2H"]
    else:
        mu = grid["C1F"]*mu + grid["C2F"]
    return mu
#%% plotting

def scatter_tend_forcing(tend, forcing, var, plot_diff=False, hue="eta", savefig=True, fname=None):
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
    p = plt.scatter(pdat[0], pdat[1], c=color.values, s=10, cmap="cool")

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

    fig.suptitle(fname)

    if savefig:
        fig.savefig(figloc + "{}_budget/scatter/{}.png".format(var, fname),dpi=300, bbox_inches="tight")

    return fig