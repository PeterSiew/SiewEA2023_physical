import xarray as xr
import numpy as np
import datetime as dt
import ipdb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from itertools import combinations

import sys; sys.path.insert(0, '/home/pyfsiew/codes/')
import tools
from importlib import reload

import fig1B_determine_how_reproducible_within_maps as fig1B

if __name__ == "__main__":


    ### Get the H500 data 
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

    ### Start the calculation
    ms_ns = [(2,1), (3,1), (4,1), (2,2), (5,1), (6,1), (3,2), (7,1), (8,1), (4,2), (9,1), (3,3), (10,1), (5,2), 
            (11,1), (12,1), (6,2), (4,3), (13,1), (14,1), (7,2), (15,1), (5,3), (16,1), (8,2), (4,4), (17,1), (18,1), (6,3), (9,2), (19,1), (20,1), (5,4), (10,2)]
    ms_ns = [(8,1), (4,2), (3,3), (9,1), (10,1), (5,2), (11,1)]
    folder = '/home/pyfsiew/codes/som/node_dates/'
    seeds = range(0,20)
    ensemble_combinations= [i for i in combinations(seeds, 2)]
    avg_corrs = {mn:[] for mn in ms_ns}
    for mn in ms_ns:
        m=mn[0]; n=mn[1]
        print(m,n)
        node_dates = {seed: np.load(folder+'node_dates_%sx%s_%s.npy'%(m,n,seed), allow_pickle='TRUE').item() for seed in seeds}
        nodes = list(node_dates[0].keys()) # the node no. is the same within ensembles
        for en_pair in ensemble_combinations:
            en1 = en_pair[0]
            en2 = en_pair[1]
            en1_node_date = node_dates[en1]
            en2_node_date = node_dates[en2]
            #highest_corr, lowest_corr = dhrr.lookfor_identical_pattern(data_weight, en1_node_date, en2_node_date)
            en1_maps = [data_weight.sel(time=node_dates[en1][node]).mean(dim='time') for node in nodes]
            en2_maps = [data_weight.sel(time=node_dates[en2][node]).mean(dim='time') for node in nodes]
            corrs = fig1B.find_best_correlations(en1_maps, en2_maps) # This returns the highest to the lowest correlations between maps
            avg_corrs[mn].append(np.mean(corrs))

    ipdb.set_trace()
    reload(fig1B); fig1B.plotting(ms_ns, avg_corrs)

def plotting(ms_ns, corrs):

    # Test all pair of combination across ensembles. 
    # If there is are strong pattern correlation (>0.9) between all maps with a pair of ensemble, then they are "identical pairs"

    plt.close()
    fig, ax1 = plt.subplots(1,1, figsize=(5,2))

    bpcolor='k'
    x = range(len(ms_ns))
    bp = ax1.boxplot(corrs.values(), positions=x, showfliers=True, widths=0.1, patch_artist=True, whis=[0,100])
    for element in ['boxes', 'whiskers', 'caps']:
        plt.setp(bp[element], color=bpcolor, lw=2) # lw=3 in Fig.1
    for box in bp['boxes']:
        box.set(facecolor=bpcolor)
    plt.setp(bp['medians'], color='white', lw=1)
    plt.setp(bp['fliers'], marker='o', markersize=0.2, markerfacecolor=bpcolor, markeredgecolor=bpcolor)
    ax1.set_xticks(x)
    ax1.set_xlabel('SOM map size')
    ax1.annotate(r'$\bf{(B)}$ Reproducibility', xy=(-0.12, 1.05), xycoords='axes fraction')
    ax1.set_ylabel('Pattern correlation')
    ax1.set_xticklabels([str(m)+'x'+str(n) for m,n in ms_ns], rotation=90)
    ax1.set_xlim(x[0]-0.5, x[-1]+0.5)
    fig_name = 'fig1B_reprodicitivity_boxplot'
    #plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300)
    plt.savefig('/home/pyfsiew/graphs/%s_%s.svg'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300)

def find_best_correlations(fixed_maps, other_maps):

    # Find the sorted highest correlations between the assigned map size and different map sizes
    # Fixed map is the 3x3 maps in list
    # Maps is other maps that can be 7x1, 8x1, 4x2, 9x1, 10x1, 5x2, 11x1 in list
    # (Fixed) This is not perfect since it delete the fixed_map immediately after the selection of the higest correlation.
    # Fixed by calculating all correlations first and identify a pair with the highest corrs, and then delete this par - then we look for the highest corr pair again in remaining maps 

    corrs = np.zeros((len(other_maps), len(fixed_maps))) # No. of rows is the length of other maps; no. of cols is the length of fixed maps (9)
    for i, other_map in enumerate(other_maps):
        for j, fixed_map in enumerate(fixed_maps):
            __, corr = tools.pattern_correlation(fixed_map,other_map)
            corrs[i,j] = corr

    highest_corrs = [] # This is the highest corr in order
    loops = min([len(other_maps), len(fixed_maps)]) 
    for i in range(0, loops):
        i_idx, j_idx = np.unravel_index(np.argmax(corrs), corrs.shape)
        highest_corrs.append(corrs[i_idx, j_idx])
        corrs_del_row = np.delete(corrs, i_idx, axis=0)
        corrs_del_col = np.delete(corrs_del_row, j_idx, axis=1)
        corrs = corrs_del_col

    return highest_corrs # Highest to the lowest. The size depends on the lenngth of other_maps (len(other_maps))

def lookfor_identical_pattern(data, en1_date, en2_date): # not used

    # If there is are strong pattern correlation (>0.9) between all maps with a pair of ensemble, then they are "identical pairs"
    nodes = [*en1_date] # en1 and en2 have the same number of nodes
    en1_maps = {n: data.sel(time=en1_date[n]).mean(dim='time') for n in nodes}
    en2_maps = {n: data.sel(time=en2_date[n]).mean(dim='time') for n in nodes}

    # Identify the pattern correlations between en1 and en2 maps.
    corrs_all = []
    for n1 in nodes:
        corrs = []
        X = en1_maps[n1].values.flatten()
        for n2 in nodes:
            Y = en2_maps[n2].values.flatten()
            corr = ((X-X.mean())*(Y-Y.mean())).sum() / (((X-X.mean())**2).sum()*((Y-Y.mean())**2).sum())**0.5
            corrs.append(corr)
        corrs_all.append(corrs)

    corrs_all = np.array(corrs_all)
    corrs_max = np.array(corrs_all).max(axis=1)  # Find the best match with another pair

    highest_pattern_corr = corrs_max.max()
    lowest_pattern_corr= corrs_max.min()

    return highest_pattern_corr, lowest_pattern_corr

