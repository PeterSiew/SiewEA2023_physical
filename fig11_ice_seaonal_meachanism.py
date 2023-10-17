from scipy import signal
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
import cartopy.crs as ccrs
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from somoclu import Somoclu
from importlib import reload
import pandas as pd
import sys
sys.path.insert(0, '/home/pyfsiew/codes/')
import tools
import create_timeseries as ct


if __name__ == "__main__":


    vars = ['cice5_BKShi', 'cice5_BKShi', 'cice5_BKShi_dynam', 'cice5_BKShi_thermo']; years = range(1980,2018)
    vars = ['piomas_BKShi', 'piomas_BKShi', 'piomas_BKShi_advect', 'piomas_BKShi_prod']; years = range(1980,2018)
    vars = ['BKSICE', 'cice5_BKSICE', 'cice5_BKSICE_dynam', 'cice5_BKSICE_thermo']; years=range(1980,2018)
    var_ratio = [100, 100, 100*24*3600, 100*24*3600]
    var_ratio = [100, 100, 1, 1]
    vars_delta = [True, True, False, False]
    datas = {}
    for i, var in enumerate(vars):
        data = tools.read_timeseries(var, months='all', remove_winter_mean=False) * var_ratio[i]
        data = data.differentiate('time', datetime_unit='D') if vars_delta[i] else data
        datas[var] = data
    #####
    years = range(1980,2021) # Up to 2021 March
    years = range(1980,2018) # Up to 2018 March
    obs_sum_ts, cice5_sum_ts, dynam_sum_ts, thermo_sum_ts = [], [], [], []
    for year in years:
        ############ Summation of thermo change and dynam change ###############
        start_date = '%s-12-01'%year
        end_date = '%s-04-01'%(year+1)
        ####
        obs_sum = datas[vars[0]].sel(time=slice(start_date, end_date)).sum(dim='time') # ice_dynam + ice_thermo = ice_total
        cice5_sum = datas[vars[1]].sel(time=slice(start_date, end_date)).sum(dim='time') # ice_dynam + ice_thermo = ice_total
        dynam_sum = datas[vars[2]].sel(time=slice(start_date, end_date)).sum(dim='time')
        thermo_sum = datas[vars[3]].sel(time=slice(start_date, end_date)).sum(dim='time')
        obs_sum_ts.append(obs_sum)
        cice5_sum_ts.append(cice5_sum)
        dynam_sum_ts.append(dynam_sum)
        thermo_sum_ts.append(thermo_sum)
    obs_sum_ts = signal.detrend(obs_sum_ts) + np.mean(obs_sum_ts)
    cice5_sum_ts = signal.detrend(cice5_sum_ts) + np.mean(cice5_sum_ts)
    dynam_sum_ts = signal.detrend(dynam_sum_ts) + np.mean(dynam_sum_ts)
    thermo_sum_ts = signal.detrend(thermo_sum_ts) + np.mean(thermo_sum_ts)

    ###
    ###
    ### Plot the total diff, summuation of dynam change and thermo change in each year
    plt.close()
    x = np.arange(len(years))
    fig1, ax1 = plt.subplots(1,1,figsize=(6,1.5))
    ax1.plot(x, obs_sum_ts, linestyle='--', color='k', lw=0.5, label='$Obs \; \Delta ICE_{total}$')
    #ax1.plot(x, cice5_sum_ts, linestyle='-', color='k', lw=1, label='CICE5 total')
    ax1.plot(x, np.array(dynam_sum_ts)+np.array(thermo_sum_ts), linestyle='-', color='k', lw=1, label='$CICE5 \; \Delta ICE_{total}$')
    ax1.bar(x, dynam_sum_ts, 0.5, linestyle='-', color='royalblue', label='$CICE5 \; \Delta ICE_{dynam}$')
    ##### If thermo and dynam have the same sign, then stack the
    idx = [i for i,ele in enumerate(dynam_sum_ts) if dynam_sum_ts[i]*thermo_sum_ts[i]>0]
    ax1.bar(x[idx], thermo_sum_ts[idx], 0.5, linestyle='-', bottom=dynam_sum_ts[idx], color='tomato', label='$CICE5 \; \Delta ICE_{thermo}$')
    # different sign, no need to stack
    idx = [i for i,ele in enumerate(dynam_sum_ts) if dynam_sum_ts[i]*thermo_sum_ts[i]<0]
    ax1.bar(x[idx], thermo_sum_ts[idx], 0.5, linestyle='-', color='tomato')
    ### Plot the correlation
    fs = 8
    xs = 0.4
    n=38
    # Correlation between circulation index and sea ice
    #corr = tools.correlation_nan(circulation_index[0:n], cice5_sum_ts)
    #ax1.annotate(r"$\rho (circulation, \Delta ICE_{total}$) = %s"%(str(round(corr,2))),xy=(xs,1.6), xycoords='axes fraction', fontsize=fs)
    # Correlation between two two ice components
    corr = tools.correlation_nan(dynam_sum_ts[0:n], thermo_sum_ts[0:n])
    ax1.annotate(r"$\rho (\Delta ICE_{thermo}, \Delta ICE_{dynam}$) = %s"%(str(round(corr,2))),xy=(xs,1.6), xycoords='axes fraction', fontsize=fs)
    # Partial correlation
    pcorr = tools.partial_correlation_new(dynam_sum_ts[0:n], circulation_index[0:n], thermo_sum_ts[0:n])
    ax1.annotate(r"$\rho (circulation, \Delta ICE_{dynam} | \Delta ICE_{thermo}$) = %s"%(str(round(pcorr,2))),xy=(xs,1.3), xycoords='axes fraction', fontsize=fs)
    pcorr = tools.partial_correlation_new(thermo_sum_ts[0:n], circulation_index[0:n], dynam_sum_ts[0:n])
    ax1.annotate(r"$\rho (circulation, \Delta ICE_{thermo} | \Delta ICE_{dynam}$) = %s"%(str(round(pcorr,2))),xy=(xs,1.45), xycoords='axes fraction', fontsize=fs)
    # Correlation betweeen ice component and total ice
    corr = tools.correlation_nan(np.array(dynam_sum_ts)+np.array(thermo_sum_ts), thermo_sum_ts)
    ax1.annotate(r"$\rho (\Delta ICE_{total}, \Delta ICE_{thermo}$) = %s"%(str(round(corr,2))),xy=(xs,1), xycoords='axes fraction', fontsize=fs)
    corr = tools.correlation_nan(np.array(dynam_sum_ts)+np.array(thermo_sum_ts), dynam_sum_ts)
    ax1.annotate(r"$\rho (\Delta ICE_{total}, \Delta ICE_{dynam}$) = %s"%(str(round(corr,2))),xy=(xs,1.15), xycoords='axes fraction', fontsize=fs)
    #corr = tools.correlation_nan(circulation_index[0:n], dynam_sum_ts)
    #corr = tools.correlation_nan(circulation_index[0:n], obs_sum_ts)
    #####
    # Setup the graph and colorbar 
    axs = [ax1]
    for ax in axs:
        for i in ['right', 'top']:
            ax.spines[i].set_visible(False)
    ax.tick_params(axis='x', which='both',length=0)
    ax.tick_params(axis='y', which='both',length=2)
    ax1.legend(bbox_to_anchor=(-0.05, 0.9), ncol=1, loc='lower left', frameon=False, columnspacing=0.5, handletextpad=0.1, labelspacing=0)
    ax1.set_ylabel('DJFM SIC\ngrowth anomaly (%)')
    ax1.set_xticks(x)
    xticklabels = [str(yr)[2:] + '/' + str(yr+1)[2:] for yr in years]
    ax1.set_xticklabels(xticklabels, rotation=90, fontsize=9)
    ax1.set_xlim(x[0]-0.5, x[-1]+0.5)
    ax1.axhline(y=0, color='lightgray', linestyle='--', linewidth=1, zorder=-1)
    ###
    fig_name = 'total_ice_change_versus_thermo_dynam_new'
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)

