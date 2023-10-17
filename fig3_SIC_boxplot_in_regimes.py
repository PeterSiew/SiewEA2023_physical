import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib; matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
import datetime as dt
import ipdb
import matplotlib
import pandas as pd

import sys
from importlib import reload
sys.path.insert(0, '/home/pyfsiew/codes/')
import tools

import create_timeseries as ct

if __name__ == "__main__":


    '''
    ### Get the sea ice data
    index = 'BKSICE' 
    indices = ['BKSICE', 'cice5_BKSICE', 'piomas_BKSICE', 'cice5_BKShi', 'piomas_BKShi'] # BKSICE (obs) has some missing data. The CICE5 and PIOMAS has no 29 Feb
    diffs = [True, True, True, True, True]
    ratios = [100, 100, 100, 100, 100]
    tss = {}
    for i, index in enumerate(indices):
        ts = tools.read_timeseries(index, months='all', remove_winter_mean=False) # Result will be the same wether we remove the mean if we do delta timeseries
        ts = ts*ratios[i]
        tss[index] = ts.differentiate('time', datetime_unit='D')  if diffs[i] else ts
    '''

    vars = ['cdr_seaice_conc', 'cice5_sic', 'piomas_aiday', 'cice5_thickness', 'piomasAxel_hiday']
    vars_delta = [True, True, True, True, True]
    var_ratio = [100, 100, 100, 100, 100] # SIC:(0-1) to %; thickness:m to cm

    # We calculate the 1) difference; 2) remove seasonal cycle and 3) the area-weighted timeseries. Find that basically equally as above
    # which mean the sequence of 1) and 2) can switch
    tss = {}
    for i, var in enumerate(vars):
        print(var)
        data = tools.read_data(var, months='all', slicing=False) * var_ratio[i]
        data = data.sel(latitude=slice(60,90)).compute()
        data = data.differentiate('time', datetime_unit='D') if vars_delta[i] else data
        data = tools.remove_seasonal_cycle_and_detrend(data,detrend=True).compute()
        # Compute the weighted-area average of the data
        # The land region should be masked. So it is the SIC weighted area within the whole box
        lat1, lat2, lon1, lon2 = 64.5, 84.5, 10.5, 99.5 # Siew et al. 2021
        lons = data.longitude.values
        lats = data.latitude.values
        weighted_ts = ct.weighted_area_average(data.values, lat1, lat2, lon1, lon2, lons, lats, lon_reverse=False, return_extract3d=False)
        weighted_ts= xr.DataArray(weighted_ts, dims=['time'], coords={'time':data.time})
        # Compute the area-average timeseries
        tss[var]=weighted_ts

    ### Get the nodes data
    folder = '/home/pyfsiew/codes/som/node_dates/'
    m,n,seed = 4,4,1
    m,n,seed = 3,3,1
    node_date= np.load(folder + 'node_dates_%sx%s_%s.npy'%(m,n,seed), allow_pickle='TRUE').item()
    nodes = [*node_date]

    ### Get the sea ice data assoicated with the node 
    data_all = {}
    ice_lag = 0

    boxplot_data = {i:[] for i in vars}
    for index in vars:
        ts = tss[index]
        for node in nodes:
            node_days = pd.to_datetime(node_date[node])
            sel_time= node_days + pd.Timedelta(days=ice_lag) # ice lags by 0 day
            # Make sure the sel_time is in there
            time_mask = np.in1d(sel_time, ts.time)
            sel_time=sel_time[time_mask]
            data = ts.sel(time=sel_time)
            # Mask the nan data so the the boxplot can be plot correcttly
            mask = ~np.isnan(data); data=data[mask]
            boxplot_data[index].append(data)
    ### Start plotting
    plt.close()
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(4.5,3)) # Just to crop out the first ax (ax1)
    x = np.arange(len(nodes))
    axs = (ax1, ax1, ax1, ax2, ax2)
    colors = ('grey', 'sandybrown', 'saddlebrown', 'sandybrown', 'saddlebrown')
    xadjust = (-0.22, 0, 0.22, -0.12, 0.12)
    for i, index in enumerate(vars):
        bp=axs[i].boxplot(boxplot_data[index], positions=x+xadjust[i], showfliers=True, widths=0.12, whis=1.5, patch_artist=True)
        for element in ['boxes', 'whiskers', 'caps']:
            plt.setp(bp[element], color=colors[i], lw=2.5) # lw=3 in Fig.1
        #for cap in bp['caps']: # Make the wiskers horizonally longer
        #   cap.set(xdata=cap.get_xdata() + (-0.02,+0.02), linewidth=4.0)
        for box in bp['boxes']:
            box.set(facecolor=colors[i])
        plt.setp(bp['medians'], color='white', lw=2)
        plt.setp(bp['fliers'], marker='o', markersize=0.2, markerfacecolor=colors[i], markeredgecolor=colors[i])
    ### Setup the graph 
    for pos in ['right', 'left', 'top', 'bottom']:
        ax1.spines[pos].set_visible(False)
        ax2.spines[pos].set_visible(False)
    n = 2.5
    for ax in (ax1, ax2):
        ax.tick_params(axis='x', which='both',length=0)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, zorder=-1, xmin=0.01, xmax=0.99)
        ax.set_ylim(-n,n)
        ax.set_xlim(x[0]-0.5,x[-1]+0.5)
        ax.set_yticks([-1,0,1])
        ax.set_xticks(x)
    ax1.set_xticklabels([])
    ax2.set_xticklabels(nodes)
    ax2.set_xlabel('Cluster')
    #ax1.set_ylabel('SIC change (%/day)', rotation='0') #ax1.yaxis.set_label_coords(-0.3,0.3)
    ax2.set_ylabel('cm/day')
    ax1.set_ylabel('%/day')
    label_names = ['Obs', 'CICE5', 'PIOMAS']
    legends = [matplotlib.patches.Patch(facecolor=colors[i],edgecolor=colors[i],label=label_names[i]) for i in range(len(label_names))]
    ax1.legend(handles=legends, bbox_to_anchor=(-0.05, 1), ncol=3, loc='lower left', frameon=False, columnspacing=1.5, handletextpad=0.6)
    ax1.annotate(r'$\bf{(A)}$ Sea ice concentration changes', xy=(-0.12, 0.99), xycoords='axes fraction', size=9)
    ax2.annotate(r'$\bf{(B)}$ Sea ice thickness changes', xy=(-0.12, 0.85), xycoords='axes fraction', size=9)
    fig_name = 'ice_in_each_regimes_boxplot'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=-0.02)
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)


def backup():

    mean_day = 5 # pentad-mean
    if mean_day==5: # This is to make sure the first pentad is from 1Dec - 5Dec. It makes no difference if we don't do pentad
        data_temp = [ts_raw.sel(time=slice('%s-11-26'%yr, '%s-03-05'%(yr+1))).coarsen(time=mean_day, boundary='trim').mean() for yr in years]
        ts_raw = xr.concat(data_temp, dim='time')
