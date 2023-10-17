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


import four_regimes as fr
import fig1A_determine_how_many_regimes_patterncorrs as fig1A


if __name__ == "__main__":

    var = 'H500_regrid_1x1'
    # Read data and remove the seasonal cycle 
    var3d = tools.read_data(var, slicing=False, months='all', limit_year=False, compute=True)
    var3d_anom = tools.remove_seasonal_cycle_and_detrend(var3d, detrend=True)
    lon1, lon2, lat1, lat2 = -60, 90, 60, 90 # Euro-Atlantic Arctic
    lon1, lon2, lat1, lat2 = -60, 90, 50, 90 # Euro-Atlantic Arctic (Holding this to 50N is important)
    data = var3d_anom.sel(longitude=slice(lon1,lon2)).sel(latitude=slice(lat1,lat2))
    # Calculation of pattern correlation requires the cos weighting
    lat_weight = np.cos(np.radians(data.latitude))  # This might be more appropriate without sqrt
    data_weight = data * lat_weight

    ms_ns = [(2,1), (3,1), (4,1), (2,2), (5,1), (6,1), (3,2), (7,1), (8,1), (4,2), (9,1), (3,3), (10,1), (5,2), 
            (11,1), (12,1), (6,2), (4,3), (13,1), (14,1), (7,2), (15,1), (5,3), (16,1), (8,2), (4,4), (17,1), (18,1), (6,3), (9,2), (19,1), (20,1), (5,4), (10,2)]
    ms_ns = [(2,1), (3,1), (4,1), (5,1), (6,1), (7,1), (8,1), (9,1), (10,1), 
            (11,1), (12,1), (13,1), (14,1), (15,1), (16,1), (17,1), (18,1), (19,1), (20,1)]
    folder = '/home/pyfsiew/codes/som/node_dates/'
    seeds = range(0,20)
    within_cluster_corrs = {s:[] for s in seeds}
    between_cluster_corrs = {s:[] for s in seeds}
    for seed in seeds:
        for m, n in ms_ns:
            print(seed, m, n)
            node_date = np.load(folder + 'node_dates_%sx%s_%s.npy'%(m,n,seed), allow_pickle='TRUE').item()
            nodes = [*node_date]
            centroid_maps = [data_weight.sel(time=node_date[node]).mean(dim='time') for node in nodes]
            # Avg pattern corrs within within clusters
            __, pattern_corrs = fr.mangitude_regime(node_date, data_weight, centroid_maps)
            avg_pattern_corrs = np.median([pc for node in nodes for pc in pattern_corrs[node]])
            # Avg/Max pattern corrs between clusters centroid
            max_pattern_corrs = fig1A.pattern_correlation_pairs(centroid_maps)
            
            within_cluster_corrs[seed].append(avg_pattern_corrs)
            between_cluster_corrs[seed].append(max_pattern_corrs)
    
    fig1A.plotting(ms_ns, within_cluster_corrs, between_cluster_corrs)



def pattern_correlation_pairs(centroid_maps):

    # Calculate the pattern correlation between all possible centroid maps
    
    nodes = range(1, len(centroid_maps)+1)
    node_combination = [i for i in combinations(nodes, 2)]

    pattern_corrs = []
    for node in node_combination:
        nodeA = node[0]
        nodeB = node[1]
        __, pattern_corr = tools.pattern_correlation(centroid_maps[nodeA-1], centroid_maps[nodeB-1])
        pattern_corrs.append(pattern_corr)

    # The average pattern corrs between cluster centroid (Here we use the raw data)
    avg_pattern_corrs = np.median(pattern_corrs)
    #max_pattern_corrs= np.max(pattern_corrs)
    max_pattern_corrs= np.quantile(pattern_corrs, 0.9)

    return max_pattern_corrs


def plotting(ms_ns, within_cluster_pc, between_cluster_pc):


    seeds = [*within_cluster_pc]
    within_mean= np.quantile(list(within_cluster_pc.values()), 0.5, axis=0)
    within_lower = np.quantile(list(within_cluster_pc.values()), 0.99, axis=0)
    within_upper = np.quantile(list(within_cluster_pc.values()), 0.01, axis=0)

    between_mean= np.quantile(list(between_cluster_pc.values()), 0.5, axis=0)
    between_lower = np.quantile(list(between_cluster_pc.values()), 0.25, axis=0)
    between_upper= np.quantile(list(between_cluster_pc.values()), 0.75, axis=0)

    ratio = within_mean / between_mean
    plt.close()
    fig, ax1 = plt.subplots(1,1, figsize=(5,2))
    x = range(len(ms_ns))
    ax1.plot(x, within_mean, marker='o', ms=0, color='royalblue', label='Within clusters')
    ax1.fill_between(x, within_lower, within_upper, alpha=0.3, fc='#2166ac')
    ax1.plot(x, between_mean, marker='o', ms=0, color='red', label='Between clusters')
    ax1.fill_between(x, between_lower, between_upper, alpha=0.3, fc='pink')
    
    idx1 = ms_ns.index((8,1))
    idx2 = ms_ns.index((11,1))
    ax1.axvspan(idx1, idx2, alpha=0.3, color='lightgray', lw=0)

    legend = ax1.legend(bbox_to_anchor=(0.5, 0.2), ncol=1, loc='lower left', frameon=False, columnspacing=1.5, handletextpad=0.6)
    ax1.set_xlabel('SOM map size (Nx1)')
    ax1.set_xticks(x)
    ax1.set_ylim(0.25,0.65)
    ax1.set_xlim(x[0], x[-1])
    ax1.set_xticklabels([str(m) for m,n in ms_ns])
    ax1.set_ylabel('Pattern correlation')
    ax1.annotate(r'$\bf{(A)}$ Similarity', xy=(-0.12, 1.05), xycoords='axes fraction', size=10)

    ax1.yaxis.label.set_color('k')

    fig_name = 'Pattern_corr_within_between_clusters'
    #plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300)
    plt.savefig('/home/pyfsiew/graphs/%s_%s.svg'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300)
