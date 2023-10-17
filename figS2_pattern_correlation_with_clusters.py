import xarray as xr
import numpy as np
import datetime as dt
import ipdb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from importlib import reload
from itertools import combinations

import four_regimes as fr
import sys; sys.path.insert(0, '/home/pyfsiew/codes/')
import tools

if __name__ == "__main__":

    var = 'H500_regrid_1x1'
    # Read data and remove the seasonal cycle 
    var3d = tools.read_data(var, slicing=False, months='all', limit_year=False, compute=True)
    var3d_anom = tools.remove_seasonal_cycle_and_detrend(var3d, detrend=True)
    lon1, lon2, lat1, lat2 = -60, 90, 60, 90 # Euro-Atlantic Arctic
    data = var3d_anom.sel(longitude=slice(lon1,lon2)).sel(latitude=slice(lat1,lat2))
    # Calculation of pattern correlation requires the cos weighting
    lat_weight = np.cos(np.radians(data.latitude))  # This might be more appropriate without sqrt
    data_weight = data * lat_weight


    folder = '/home/pyfsiew/codes/som/node_dates/'
    m, n = 3,3 
    seed = 1
    node_date = np.load(folder + 'node_dates_%sx%s_%s.npy'%(m,n,seed), allow_pickle='TRUE').item()
    nodes = [*node_date]
    centroid_maps = [data_weight.sel(time=node_date[node]).mean(dim='time') for node in nodes]
    # Avg pattern corrs within within clusters
    __, pattern_corrs = fr.mangitude_regime(node_date, data_weight, centroid_maps)


    # ipdb.set_trace()
    plt.close()
    fig, ax1 = plt.subplots(1,1, figsize=(5,2))
    x = nodes
    bp1 = ax1.boxplot(list(pattern_corrs.values()), positions=x, showfliers=False, widths=0.1)
    ax1.set_xlabel('Clusters')
    ax1.set_ylabel('Pattern\ncorrelations')
    ax1.set_ylim(-1,1)
    ax1.axhline(y=0, color='lightgray', linestyle='--', linewidth=1, zorder=-1)
    fig_name = 'regime_pattern_correlations'
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300)
