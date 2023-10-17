import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
from scipy import signal
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from importlib import reload
from scipy import stats

import sys; sys.path.insert(0, '/home/pyfsiew/codes/')
import tools

import fig10_frequency_cluster_seasonal_ice as fig10


if __name__ == "__main__":

    # Count the circulation clusters ahead of them in Days 5, 10, 15 and 20
    folder = '/home/pyfsiew/codes/som/node_dates/'
    m,n,seed = 3,3,1
    node_date= np.load(folder + 'node_dates_%sx%s_%s.npy'%(m,n,seed), allow_pickle='TRUE').item()
    nodes = [*node_date]
    # Create a dictionary of date: regime from node_date
    node_timeseries = {time:node for node in nodes for time in node_date[node]}

    if False: ### Get the circulation mangitude using the dot product
        var = 'H500_regrid_1x1'
        data = tools.read_data(var, slicing=False, months='all', limit_year=False, compute=True)
        data = tools.remove_seasonal_cycle_and_detrend(data, detrend=True)
        lon1, lon2, lat1, lat2 = -60, 90, 60, 90 
        data = data.sel(longitude=slice(lon1,lon2)).sel(latitude=slice(lat1,lat2))
        lat_weight = np.sqrt(np.cos(np.radians(data.latitude)))  
        data_weight = data*lat_weight
        #data_weight= data
        centroid_maps = [data_weight.sel(time=node_date[node]).mean(dim='time') for node in nodes]
        node_mags_list = []
        if True:
            cluster_mags = {}
            for node in nodes:
                node_data = data_weight.sel(time=pd.to_datetime(node_date[node]))
                node_centroid = centroid_maps[node-1]
                tdim=node_data.time.size
                node_mag = np.dot(node_data.values.reshape(tdim,-1), node_centroid.values.reshape(-1))
                node_mag = (node_mag-node_mag.min()) / (node_mag.max()-node_mag.min()) # Noramize between 0 and 1
                for i,day in enumerate(node_date[node]):
                    cluster_mags[day]=node_mag[i]
        else:
            cluster_mags = {day:tools.pattern_correlation(data.sel(time=day),centroid_maps[node-1])[0] for node in nodes for day in node_date[node]}

    # Get the sea ice timeseries. Get the extreme sea ice events. 
    index = 'BKSICE_raw'
    ice_raw = tools.read_timeseries(index, months='all', remove_winter_mean=False) * 100
    #ice_anom = tools.remove_seasonal_cycle_and_detrend(ice_raw, detrend=False) # Create almost the same result
    ice_diff = ice_raw.differentiate('time', datetime_unit='D') 
    years = range(1980,2021) # up to 2021 Mar
    node_in_years = {yr: {node:0 for node in nodes} for yr in years}
    mag_in_years = {yr: {node:[] for node in nodes} for yr in years}
    winter_ices = []
    autumn_ices= []
    ice_diffs = [] 
    ice_diff_sum = []
    for i, yr in enumerate(years):
        winter_time = pd.date_range(start='%s-12-01'%(yr), end='%s-03-31'%(yr+1), freq='D') # year 1983 means 1983-12-01 to 1984-03-31
        for day in winter_time:
            node = node_timeseries[day]
            node_in_years[yr][node] += 1
            #mag_in_years[yr][node].append(cluster_mags[day])
            mag_in_years[yr][node].append(1)
        winter_ice = ice_raw.sel(time=winter_time).mean(dim='time').values.item()
        winter_ices.append(winter_ice)
        ### The mean autumn season
        autumn_time = pd.date_range(start='%s-11-01'%yr, end='%s-11-30'%yr, freq='D') 
        autumn_ice = ice_raw.sel(time=autumn_time).mean(dim='time').values.item()
        autumn_ices.append(autumn_ice)
        ### The diff between the end of a season and the start of a season
        winter_st = pd.date_range(start='%s-11-29'%yr, end='%s-12-03'%yr, freq='D')
        ice_st = ice_raw.sel(time=winter_st).mean(dim='time').values.item()
        winter_end = pd.date_range(start='%s-03-29'%(yr+1), end='%s-03-31'%(yr+1), freq='D')
        ice_end = ice_raw.sel(time=winter_end).mean(dim='time').values.item()
        ice_diffs.append(ice_end-ice_st)
        #### Define the change first, and the summation of changes across the season
        start_date = '%s-12-01'%yr
        end_date = '%s-04-01'%(yr+1)
        obs_sum = ice_diff.sel(time=slice(start_date, end_date)).sum(dim='time').item() # ice_dynam + ice_thermo = ice_total
        ice_diff_sum.append(obs_sum)
    winter_ices = signal.detrend(winter_ices) + np.mean(winter_ices)
    autumn_ices = signal.detrend(autumn_ices) + np.mean(autumn_ices)
    #seasonal_diffs = np.array(winter_ices)-np.array(autumn_ices)
    #seasonal_diffs = signal.detrend(ice_diffs) + np.mean(ice_diffs)
    seasonal_diffs = signal.detrend(ice_diff_sum) + np.mean(ice_diff_sum)
    ############################################
    ############################################
    ### Start plotting
    plt.close()
    fig, ax1 = plt.subplots(1,1,figsize=(6,3))
    plotting_bars = {node:[node_in_years[yr][node] for yr in years] for node in nodes}
    ### Plot the bar (the frequency of clusters)
    x = range(len(years))
    bottom = 0
    bar_width = 0.7
    node_col= {1:'#2166ac', 4:'#d1e5f0', 7:'#f4a582', 2:'#4393c3', 5:'#969696', 8:'#d6604d', 3:'#92c5de', 6:'#fddbc7', 9:'#b2182b'}
    node_col= {9:'#2166ac', 6:'#d1e5f0', 3:'#f4a582', 8:'#4393c3', 5:'#969696', 2:'#d6604d', 7:'#92c5de', 4:'#fddbc7', 1:'#b2182b'}
    for i, node in enumerate(nodes[::-1]):
        ax1.bar(x, plotting_bars[node], bar_width, bottom=bottom, color=node_col[node])
        bottom = bottom + np.array(plotting_bars[node])
    ax1.set_xticks(x)
    xticklabels = [str(yr)[2:] + '/' + str(yr+1)[2:] for yr in years]
    ax1.set_xticklabels(xticklabels, rotation=90, fontsize=8)
    ax1.set_xlim(x[0]-0.5, x[-1]+0.5)
    #ax1.set_title('(B)', loc='left')
    ax1.set_ylabel('Counts',labelpad=-0.5)
    #Set up ax2
    ax2 = ax1.twinx()
    ax2.plot(x, winter_ices, linestyle='-', color='k', lw=2)
    ax2.plot(x, seasonal_diffs, linestyle='--', color='k', lw=2)
    ax2.axhline(y=0, color='lightgray', linestyle='--', linewidth=1, zorder=-1)
    ax2.set_ylabel('DJFM seasonal mean SIC \n and its grwoth (%)')
    ax2.set_ylim(0,30)
    ax2.set_ylim(0,75)
    ### Plot the circulation index (gray line)
    # setting_ratio
    plotting_bars_mag = {node:[np.mean(mag_in_years[yr][node]) for yr in years] for node in nodes}
    ratio = [-0.469, -0.403, -0.272, -0.222, 0.082, 0.202, 0.214, 0.439, 0.389] # This is not very useful
    ratio = [-1, -1, -1, -1, 0, 1, 1, 1, 1] 
    for i, node in enumerate(nodes):
        node_mag = np.array(plotting_bars_mag[node])
        node_mag = np.where(np.isnan(node_mag),0,node_mag) #if a year has no that node, it appears as nan
        plotting_bars[node] = np.array(plotting_bars[node]) * ratio[i]
        plotting_bars[node] = np.array(plotting_bars[node]) * node_mag
    # Set the the correlation
    #circulation_index = high_ice_regimes + low_ice_regimes
    circulation_index = plotting_bars[1] + plotting_bars[2] + plotting_bars[3] + plotting_bars[4] + plotting_bars[5] + \
                        plotting_bars[6] + plotting_bars[7] + plotting_bars[8] + plotting_bars[9]
    corr_normal = tools.correlation_nan(winter_ices, circulation_index)
    corr_diff  = tools.correlation_nan(seasonal_diffs, circulation_index)
    ax2.annotate(r"$\rho$ = %s (DJFM mean)"%(str(round(corr_normal,2))),xy=(0.6,1.05), xycoords='axes fraction')
    ax2.annotate(r"$\rho$ = %s (DJFM growth)"%(str(round(corr_diff,2))),xy=(0.6,0.98), xycoords='axes fraction')
    # Plot the circulation_index
    ax3 = ax1.twinx()
    ax3.plot(x, circulation_index, linestyle='-', color='grey', lw=1)
    ax3.set_yticks([])
    ax3.set_yticklabels([])
    ax3.yaxis.set_ticks_position('none') 
    # Setup the graph and colorbar 
    axs = [ax1, ax2, ax3]
    for ax in axs:
        for i in ['left', 'right', 'top']:
            ax.spines[i].set_visible(False)
        ax.tick_params(axis='x', which='both',length=1)
        ax.tick_params(axis='y', which='both',length=2)
    ### Set up legend
    legends = [matplotlib.patches.Patch(facecolor=node_col[node],edgecolor=node_col[node],label=node) for node in nodes]
    legends = np.array(legends).reshape(3,3).T.flatten().tolist()
    ax1.legend(handles=legends, bbox_to_anchor=(0.2,0.9), ncol=3, loc='lower left', frameon=False, columnspacing=0.5, handletextpad=-1.2, labelspacing=-0.15)
    fig_name = 'cluster_freq_seasonal_ice'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.5)
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)

    if False: ### Do the multivariate regression
        import statsmodels.api as sm
        #predict_y = np.column_stack(winter_ices).T
        predict_y = np.column_stack(seasonal_diffs).T
        predict_x = np.column_stack([plotting_bars[1], plotting_bars[2],plotting_bars[3],plotting_bars[4],
                        plotting_bars[5],plotting_bars[6],plotting_bars[7],plotting_bars[8],plotting_bars[9]])
        predict_y = (predict_y - predict_y.mean()) / predict_y.std()
        betas, residual = tools.multiple_linear_regression(predict_y, predict_x)
        # Use SM method
        predict_x= sm.add_constant(predict_x)
        regress_results = sm.OLS(endog=predict_y, exog=predict_x).fit()
        beta_coeffs = regress_results.params
        pvals = regress_results.pvalues

