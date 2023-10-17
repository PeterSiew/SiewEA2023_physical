import xarray as xr
import numpy as np
import ipdb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
import cartopy.crs as ccrs
import datetime as dt
from importlib import reload
from scipy import stats
import pandas as pd


import sys; sys.path.insert(0, '/home/pyfsiew/codes/')
import tools
import fig9_total_ice_change_persist_days as fig9



if __name__ == "__main__":

    ##########################################################################
    ### Gettting the mode_date
    folder = '/home/pyfsiew/codes/som/node_dates/'
    m,n,seed = 3,3,1
    mode_date = np.load(folder + 'node_dates_%sx%s_%s.npy'%(m,n,seed), allow_pickle='TRUE').item()
    modes = [*mode_date]

    if False: ### Get the circulation mangitude (We don't need that here)
        var = 'H500_regrid_1x1'
        var3d = tools.read_data(var, slicing=False, months='all', limit_year=False, compute=True)
        var3d = tools.remove_seasonal_cycle_and_detrend(var3d, detrend=True)
        var3d = var3d.sel(longitude=slice(-60,90)).sel(latitude=slice(60,90))
        cluster_centroids = {mode: var3d.sel(time=mode_date[mode]).mean(dim='time') for mode in modes}
        cluster_mags = {day: tools.pattern_correlation(var3d.sel(time=day),cluster_centroids[mode])[0] for mode in modes for day in mode_date[mode]}

    ### Select the mode date
    iceloss_dates = sorted(mode_date[1]+mode_date[2]+mode_date[3]+mode_date[4])
    icegrowth_dates= sorted(mode_date[6]+mode_date[7]+mode_date[8]+mode_date[9])
    titles=['Ice-loss patterns', 'Ice-growth patterns']
    ABCtitles=['A', 'B']
    if False: ### Get the random data
        corrs = []
        multiple_mode_datess = sorted([date for m in modes for date in mode_date[m]]); ABCtitle='C'; title='Bootstrap samples'
        random_idx = np.random.choice(len(multiple_mode_datess), size=2198, replace=False)
        multiple_mode_dates = sorted(list(np.array(multiple_mode_datess)[random_idx]))

    ### Get the persistent day details
    days = range(1,40)
    iceloss_persist_date = {d:None for d in days} # key: persist day, mode
    icegrowth_persist_date = {d:None for d in days} # key: persist day, mode
    for day in days:
        iceloss_persist_date[day]= fig9.persistent_count(iceloss_dates, day, next_day_interval=1)
        icegrowth_persist_date[day]= fig9.persistent_count(icegrowth_dates, day, next_day_interval=1)


    # Check the total number # total_no = [len(persist_date[d]) for d in days]
    def getting_plotting_data(ts, ice_persist_date, dot_color='k'):
        ### Get the persistence and circulation index
        day_col = {d:dot_color for d in days}
        xs, ys, cs = [], [], []
        start_date, last_date = [], []
        for day in days:
            dates = ice_persist_date[day]
            for element in dates:
                #element_mag = [cluster_mags[ele] for ele in element] # Do the summation of regression index when the circulation persist
                # Only consider the lag days but not mangitude
                element_mag = np.repeat(1, len(element)) 
                add_time = [element[-1] + pd.Timedelta(days=d) for d in [0]] # Add one more days give the highest correlation (put 0 for add no day)
                sel_time = pd.to_datetime(element + add_time) # Add two list together
                if ~np.all(np.in1d(sel_time, ts.time)):  # Only proceed if all days are there in data (without all the data - we cannot get the summuation of changes
                    continue
                ice_anom = ts.sel(time=sel_time).sum(dim='time').item() # The summuation of change is the same as the ice (last day) - ice (first day)
                xs.append(np.sum(element_mag))
                ys.append(ice_anom)
                cs.append(day_col[day])
                start_date.append(sel_time[0])
                last_date.append(sel_time[-1])
        return xs, ys, cs, start_date, last_date

    ### Get ice data 
    ts_ratio = {'BKSICE':100, 'cice5_BKSICE':100, 'cice5_BKSICE_dynam':1, 'cice5_BKSICE_thermo':1, 'cice5_BKShi':100}
    # read ice change
    index = 'BKSICE'
    delta_diff = True
    ts = tools.read_timeseries(index, months='all', remove_winter_mean=False) * ts_ratio[index]
    ts = ts.differentiate('time', datetime_unit='D') if delta_diff else ts
    ice_change = ts
    xs1, ys1, cs1, st_date1, end_date1 = getting_plotting_data(ice_change, iceloss_persist_date, dot_color='k')
    xs2, ys2, cs2, st_date2, end_date2 = getting_plotting_data(ice_change, icegrowth_persist_date, dot_color='k')
    iceloss_corr  = tools.correlation_nan(xs1, ys1)
    icegrowth_corr = tools.correlation_nan(xs2, ys2)
    corrs = [iceloss_corr, icegrowth_corr]

    if True: ### Start the plotting
        plt.close()
        fig, (ax1, ax2) = plt.subplots(2,1,figsize=(3,3))
        ss = 1
        ax1.scatter(xs1, ys1, c='black', s=ss, marker='o')
        ax2.scatter(xs2, ys2, c='black', s=ss, marker='o')
        ax1.set_xticklabels([])
        for i, ax in enumerate([ax1, ax2]):
            ax.set_ylabel('Total SIC\nchange (%)') #ax1.yaxis.set_label_coords(-0.18,0.5)
            ax.axhline(y=0, color='lightgray', linestyle='--', linewidth=1, zorder=-1)
            ax.annotate(r"$\rho$= %s" %(str(round(corrs[i],2))), xy=(0.7, 0.9), xycoords='axes fraction')
            ax.annotate(r'$\bf{(%s)}$ %s'%(ABCtitles[i], titles[i]), xy=(-0.27, 1.05), xycoords='axes fraction', size=11)
            for pos in ['right', 'top']:
                ax.spines[pos].set_visible(False)
        ax2.set_xlabel('Persistence (day)')
        ax1.set_yticks([-15,-10,-5,0,5]); ax1.set_ylim(-17,8)
        ax2.set_yticks([-5,0,5,10,15]); ax2.set_ylim(-6,14)
        ### Save the figure
        fig_name = 'ice_anom_persist'
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.3)
        plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500,pad_inches=0.01) 

    if False: # the Fig9C (ax1 - the the result from the bootstrapping test)
        plt.close()
        fig, ax1 = plt.subplots(1,1,figsize=(3,3))
        corrs_combine = [corrs_ice_loss, corrs_ice_growth]
        x = [1,2]
        bp = ax1.boxplot(corrs_combine, positions=x, showfliers=False, widths=0.1, patch_artist=True)
        bpcolor='k'
        for element in ['boxes', 'whiskers', 'caps']:
            plt.setp(bp[element], color=bpcolor, lw=2) # lw=3 in Fig.1
        for box in bp['boxes']:
            box.set(facecolor=bpcolor)
        plt.setp(bp['medians'], color='white', lw=1)
        plt.setp(bp['fliers'], marker='o', markersize=0.2, markerfacecolor=bpcolor, markeredgecolor=bpcolor)
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Ice-loss\npatterns', 'Ice-growth\npattens'])
        ax1.set_ylabel('Correlations')
        ax1.set_xlim(x[0]-0.5, x[-1]+0.5)
        ax1.axhline(y=0, color='grey', linestyle='--', linewidth=1, zorder=-1, xmin=0.01, xmax=0.99)
        for pos in ['right', 'top']:
            ax1.spines[pos].set_visible(False)
        ax1.annotate(r'$\bf{(C)}$', xy=(-0.23, 0.95), xycoords='axes fraction', size=10)
        fig_name = 'ice_growth_bootstrap'
        plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300)


def persistent_count(regime_dates, persist_day, next_day_interval=1):

    # The regime_dates should be sorted
    # This is a problem (the last day of each mode cannot be added). Adding a 'fake' day solve the problem
    extra_date = regime_dates[-1] + pd.Timedelta(days=9999)
    regime_dates = regime_dates + [extra_date]

    regime_dates_new= []
    count=1; days_temp = []
    for d, day in enumerate(regime_dates[0:-1]): 

        # Check if the current day + 1 day is equal to the next element
        next_day = day + pd.Timedelta(days=next_day_interval) # 5 for pentad, #1 for daily
        days_temp.append(day)
        if next_day == regime_dates[d+1]:
            #days_temp.append(next_day) # This is not necessary because the 'next day' will be added in the next loop
            count = count + 1
        else: # the continuous regime situation breaks
            if persist_day-1 < count < persist_day+1: 
                regime_dates_new.append(days_temp)
            count = 1; days_temp = []

    return regime_dates_new
