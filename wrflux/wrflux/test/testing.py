#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 09:35:43 2020

@author: c7071088
"""

import tools

def test_tendencies(datout):
    base = "cartesian"
    for correct in [False, True]:
        for recalc_w in [False, True]:
            for sum_dir in [False, True]:
                ID = base
                if correct:
                    ID += " correct"
                if recalc_w:
                    ID += " recalc_w"
                ID2 = ID +" 2nd"
                if ID not in datout["adv"].ID or ID2 not in datout["adv"].ID:
                    continue

                dat = datout["adv"]
                sum_s = ""
                if sum_dir:
                    dat = dat.sel(dir="sum")
                    sum_s = "sum"
                ref = dat.sel(ID=ID)
                dat = dat.sel(ID=ID2)
                # ref = ["flux"].Z.sel(ID=ID)
                # dat = ["flux"].Z.sel(ID=ID2)
                e = tools.max_error_scaled(dat, ref)*100
                plot_kws = dict(
                    # plot_diff = True, #plot difference between forcing and tendency against tendency
                    # discrete=True,
                    # iloc={"y" : slice(15,None)},
                    # iloc={"bottom_top_stag" : slice(1,2), "x" : slice(5,7),  "y" : [0,1,9,10,11,-2,-1]},
                    # loc=None,
                    # hue="y",
                    s=10,
                    )
                thresh = 0.5
                if e > thresh:
                    log = " '{}' vs '{}': maximum error in adv {} more than {}%: {:.3f}%".format(ID, ID2, sum_s, thresh, e)
                    print(log)
                    tools.scatter_hue(dat, ref, title=log, **plot_kws)
#%%native sum vs. cartesian sum
    ID = "native"
    ID2 = "cartesian correct recalc_w"
    data = datout["adv"].sel(dir="sum")

    ref = data.sel(ID=ID)
    dat = data.sel(ID=ID2)

    e = tools.max_error_scaled(dat, ref)*100
    plot_kws = dict(
        # s=5,
        # discrete=True,
        # plot_diff=True,
        # iloc= {"x" : slice(1,5)},
        # iloc= {"bottom_top" : slice(0,5)},
        hue="comp",
         # plot_iloc =
    )
    thresh = 1 #TODO really need such high threshold? problem with mean?
    if e > thresh:
        log = " '{}' vs '{}': maximum error in adv sum more than {}%: {:.3f}%".format(ID, ID2, thresh, e)
        print(log)
        tools.scatter_hue(dat, ref, title=log, **plot_kws)

#%%recalc_w vs not recalc_w
    base_i = base + " correct"
    ID = base_i + " recalc_w"
    ID2 = base_i
    for sum_dir in [False]:
        data = datout["adv"]
        if sum_dir:
            data = data.sel(dir="sum")
            thresh = 2 #TODO really need such high threshold? problem with mean?
            sum_s = "sum"
        else:
            thresh = 1
            sum_s = ""
        ref = data.sel(ID=ID)
        dat = data.sel(ID=ID2)
        # dat = dat_mean["FQZ_ADV_MEAN"]
        # ref = dat_mean["FQZ_ADV_MEAN_PROG"]

        e = tools.max_error_scaled(dat, ref)*100
        plot_kws = dict(
            s=20,
            # discrete=True,
            # plot_diff=True,
            # iloc= {"x" : slice(1,5)},
            # iloc= {"bottom_top" : slice(0,5)},
            hue="comp",
             # plot_iloc =
        )
        if e > thresh:
            log = " '{}' vs '{}': maximum error in adv {} more than {}%: {:.3f}%".format(ID, ID2, sum_s, thresh, e)
            print(log)
            tools.scatter_hue(dat, ref, title=log, **plot_kws)


#%% mean vs adv_r
    # for ID in adv.ID:
    #     dat = adv.sel(ID=ID, comp="mean")
    #     ref = adv.sel(ID=ID, comp="adv_r")
    #     e = tools.max_error_scaled(dat, ref)*100 #TODO use scaled rmse?
    #     thresh = 15
    #     if e > thresh:
    #         log = " mean vs adv_r: maximum difference more than {}%: {:.3f}% ; for ID: {}".format(thresh, e, ID.values)
    #         print(log)
    #         tools.scatter_hue(dat, ref, hue="dir", title=log)
    # adv_sum = adv.sum("dir")
    # adv_sum[:,:,:,-1,[0,2,5]].plot(hue="comp",row="ID", y="bottom_top", col="x")
    # adv[-1,:,:,:,-1,[0,2,5]].plot(hue="comp", row="dir", y="bottom_top", col="x")
    #TODO: why is turbulent so strong on slope?
#%% total tendencies with/without correction
    ID = "cartesian correct 2nd"
    ID2 = "native"
    dat = datout["tend"][0].sel(ID=ID)
    ref = datout["tend"][0].sel(ID=ID2)
    e = tools.max_error_scaled(dat, ref)*100
    thresh = 3
    if e > thresh:
        log = " '{}' vs '{}': maximum difference in tend more than {}%: {:.3f}%".format(ID, ID2, thresh, e)
        print(log)
        tools.scatter_hue(dat, ref, hue="bottom_top", title=log)


def test_w(dat_inst):
    #%% w vs wd
    dat = dat_inst["W"]
    ref = dat_inst["W_DIAG"]
    # dat = dat_mean["W_MEAN"]
    # ref = dat_mean["WD_MEAN"]
    e = tools.max_error_scaled(dat, ref)*100
    thresh = 0.5
    plot_kws = dict(
    # s=5,
    # discrete=True,
    # plot_diff=True,
    # iloc= {"x" : slice(1,5), "y" : slice(2,4)},
    # iloc= {"y" : slice(0,5)},
    # hue="x",
     # plot_iloc =
     )
    if e > thresh:
        log = " w vs wd: maximum difference more than {}%: {:.3f}%".format(thresh, e)
        print(log)
        tools.scatter_hue(dat, ref, title=log, **plot_kws)

    dat, ref = dat.sel(bottom_top_stag=1), ref.sel(bottom_top_stag=1)
    e = tools.max_error_scaled(dat, ref)*100
    thresh = 0.5
    if e > thresh:
        log = " w vs wd at k=1: maximum difference more than {}%: {:.3f}%".format(thresh, e)
        print(log)
        tools.scatter_hue(dat, ref, hue="x", title=log)

