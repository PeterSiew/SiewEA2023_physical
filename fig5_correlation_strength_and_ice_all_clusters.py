import xarray as xr
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib; matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
import pandas as pd
import ipdb
from importlib import reload
import scipy

import four_regimes as fr
import sys; sys.path.insert(0, '/home/pyfsiew/codes/')
import tools
import create_timeseries as ct


import four_regimes as fr

if __name__ == "__main__":


    ### Get the circualtion strength data
    var = 'H500_regrid_1x1'
    # Read data and remove the seasonal cycle 
    var3d = tools.read_data(var, slicing=False, months='all', limit_year=False, compute=True)
    var3d_anom = tools.remove_seasonal_cycle_and_detrend(var3d, detrend=True)
    lon1, lon2, lat1, lat2 = -60, 90, 60, 90 # Euro-Atlantic Arctic
    lon1, lon2, lat1, lat2 = -60, 90, 65, 90 # Euro-Atlantic Arctic
    data = var3d_anom.sel(longitude=slice(lon1,lon2)).sel(latitude=slice(lat1,lat2))
    # Calculation of pattern correlation requires the cos weighting
    lat_weight = np.sqrt(np.cos(np.radians(data.latitude)))  # This might be more appropriate without weighting
    #data_weight = data # We don't need to weight the data. If weight, do data * lat_weight
    data_weight = data*lat_weight
    # Get the node data 
    folder = '/home/pyfsiew/codes/som/node_dates/'
    m,n = 3,3; seed=1
    node_date = np.load(folder + 'node_dates_%sx%s_%s.npy'%(m,n,seed), allow_pickle='TRUE').item()
    nodes = [*node_date]
    if True: # The centroid defined by the composite maps
        centroid_maps = [data_weight.sel(time=node_date[node]).mean(dim='time') for node in nodes]
    else: # The real centroid
        centroid_maps = xr.open_dataset(folder+'node_centroids_%sx%s_%s.nc'%(m,n,seed))
        centroid_maps = centroid_maps.sel(longitude=slice(lon1,lon2)).sel(latitude=slice(lat1,lat2))
        centroid_maps = [centroid_maps.isel(nodes=node-1).to_array().squeeze() for node in nodes]
    #regressions, pattern_corrs = fr.mangitude_regime(node_date, data_weight, centroid_maps)

    if True: # use to dot product to define the mangitude
        regressions = {}
        for node in nodes:
            node_data = data_weight.sel(time=pd.to_datetime(node_date[node]))
            node_centroid = centroid_maps[node-1]
            tdim=node_data.time.size
            print(tdim)
            node_mag=np.dot(node_data.values.reshape(tdim,-1), node_centroid.values.reshape(-1))
            node_mag = (node_mag-node_mag.min()) / (node_mag.max()-node_mag.min()) # Noramize between 0 and 1
            regressions[node]=node_mag

    ''' # This gives a slightly weaker correlation. But more or less the same
    vars = ['cdr_seaice_conc', 'cice5_sic', 'cice5_thickness', 'piomas_aiday', 'piomasAxel_hiday']
    tss = {}
    for i, var in enumerate(vars):
        print(var)
        data = tools.read_data(var, months='all', slicing=False) * ratios[i]
        data = data.sel(latitude=slice(60,90)).compute()
        data = data.differentiate('time', datetime_unit='D') if diffs[i] else data
        data = tools.remove_seasonal_cycle_and_detrend(data,detrend=True).compute()
        # Compute the weighted-area average of the data
        lat1, lat2, lon1, lon2 = 64.5, 84.5, 10.5, 99.5 # Siew et al. 2021
        lons = data.longitude.values
        lats = data.latitude.values
        weighted_ts = ct.weighted_area_average(data.values, lat1, lat2, lon1, lon2, lons, lats, lon_reverse=False, return_extract3d=False)
        weighted_ts= xr.DataArray(weighted_ts, dims=['time'], coords={'time':data.time})
        # Compute the area-average timeseries
        tss[indices[i]]=weighted_ts
    '''
    ### Get the sea ice data
    indices = ['BKSICE', 'cice5_BKSICE', 'cice5_BKShi', 'piomas_BKSICE', 'piomas_BKShi']
    var_name = ['NSIDC SIC', 'CICE5 SIC', 'CICE5 thickness', 'PIOMAS SIC', 'PIOMAS thickness']
    bar_colors = ['grey', 'sandybrown', 'sandybrown', 'saddlebrown', 'saddlebrown']
    hatches = [None, None, 'xxxx', None, 'xxxx']
    edge_colors= [None, None, 'k', None, 'k']
    diffs = [True, True, True, True, True]
    ratios = [100, 100, 100, 100, 100]
    tss = {}
    for i, index in enumerate(indices):
        # Result will be the same wether we remove the mean if we do delta timeseries
        ts = tools.read_timeseries(index, months='all', remove_winter_mean=False) #
        ts = ts*ratios[i]
        tss[index] = ts.differentiate('time', datetime_unit='D')  if diffs[i] else ts
    ### Caclulation the correlation
    corrs = {node: {var:None for var in indices} for node in nodes}
    ice_tss = {node: {var:None for var in indices} for node in nodes}
    circulation_strengths = {node: {var:None for var in indices} for node in nodes}
    for node in nodes:
        for var in indices:
            # Can also include lead-lag here
            circulation_strength = regressions[node]
            # To make sure the node_date[node] are all available in the ice timeseresi
            time_mask = np.in1d(pd.to_datetime(node_date[node]), tss[var].time)
            circulation_strength = np.array(circulation_strength)[time_mask]
            ice_ts = tss[var].sel(time=np.array(node_date[node])[time_mask])
            # Get the correlation
            corr = tools.correlation_nan(ice_ts, circulation_strength) # The correlations between them without lead-lag
            # Get the critical correlation
            nn=len(ice_ts)
            corrs[node][var] = corr
            ice_tss[node][var] = ice_ts
            circulation_strengths[node][var] = circulation_strength
    ### Start plotting
    plt.close()
    plt.figure(figsize=(7, 4))
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,2)
    ax3 = plt.subplot(2,1,2)
    ### For ax 1 and 2
    node = 1
    ms = 3
    corr_x=0.55; corr_y=1.1; fs=10
    colors = ['grey', 'sandybrown', 'saddlebrown']
    corr_names = ['Obs', 'CICE5', 'PIOMAS']
    for i, var in enumerate(['BKSICE', 'cice5_BKSICE', 'piomas_BKSICE']):
        ax1.scatter(circulation_strengths[node][var], ice_tss[node][var], color=colors[i], marker='o', s=ms)
        corr = round(corrs[node][var],2)
        ax1.annotate(r"$\rho$=%s (%s)"%(corr,corr_names[i]), xy=(corr_x, corr_y-i*0.1), xycoords='axes fraction', fontsize=fs, color=colors[i])
    colors = ['sandybrown', 'saddlebrown']
    corr_names = ['CICE5', 'PIOMAS']
    for i, var in enumerate(['cice5_BKShi', 'piomas_BKShi']):
        ax2.scatter(circulation_strengths[node][var], ice_tss[node][var], color=colors[i], marker='o', s=ms)
        corr = round(corrs[node][var],2)
        ax2.annotate(r"$\rho$=%s (%s)"%(corr,corr_names[i]), xy=(corr_x, corr_y-i*0.1), xycoords='axes fraction', fontsize=fs, color=colors[i])
    for ax in [ax1, ax2]:
        ax.axhline(y=0, color='lightgray', linestyle='--', linewidth=1, zorder=0)
        ax.set_xlabel('Circulation strength')
    ax1.set_ylabel('SIC change \n (%/day)')
    ax2.set_ylabel('Thickness change \n (cm/day)')
    ### For the ax 3
    xs = np.arange(1, len(nodes)+1)
    barwidth = 0.15
    x_adjust = np.linspace(-0.3,0.3,len(indices))
    for v, var in enumerate(indices):
        ax3.bar(xs+x_adjust[v], [corrs[node][var] for node in nodes], barwidth, label='%s'%var_name[v], color=bar_colors[v], edgecolor=edge_colors[v], hatch=hatches[v])
    ax3.set_xticks(xs)
    ax3.set_xticklabels(nodes)
    ax3.axhline(y=0, color='lightgray', linestyle='--', linewidth=1)
    ax3.set_ylim(-0.6,0.65)
    ax3.set_yticks([-0.6,-0.3,0,0.3,0.6])
    ax3.set_ylabel('Correlations')
    ax3.set_xlabel('Cluster')
    ax3.legend(bbox_to_anchor=(0,1.15), ncol=3, loc='upper left', frameon=False, columnspacing=1, handletextpad=0.4)
    for pos in ['right', 'top']:
        for ax in [ax1, ax2, ax3]:
            ax.spines[pos].set_visible(False)
    ax1.annotate(r'$\bf{(A)}$', xy=(-0.23, 1.1), xycoords='axes fraction', size=10)
    ax2.annotate(r'$\bf{(B)}$', xy=(-0.23, 1.1), xycoords='axes fraction', size=10)
    ax3.annotate(r'$\bf{(C)}$', xy=(-0.09, 1.1), xycoords='axes fraction', size=10)
    # Add the significant line
    nn=500 # Each node has around 500 samples
    tt=scipy.stats.t.isf(0.025, nn, loc=0, scale=1); critical_corr=tt/(nn+tt**2)**0.5
    ax3.axhline(y=critical_corr, color='black', linestyle='--', linewidth=0.5, zorder=-1)
    ax3.axhline(y=-critical_corr, color='black', linestyle='--', linewidth=0.5, zorder=-1)
    ####
    fig_name = 'correlations_circulation_strength_and_seaice'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.6) # hspace is the vertical
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)



