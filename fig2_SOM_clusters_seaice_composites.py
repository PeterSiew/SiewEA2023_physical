import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datetime as dt
import ipdb
import pandas as pd
import cartopy.crs as ccrs
matplotlib.rcParams['font.sans-serif'] = "URW Gothic L"
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"

from importlib import reload
import sys; sys.path.insert(0, '/home/pyfsiew/codes/')
import tools
from scipy import stats

import fig2_SOM_clusters_seaice as fig2


if __name__ == "__main__":

    vars = ['H500_regrid_1x1', 'cdr_seaice_conc', 'U10M', 'V10M']
    vars = ['H500', 'cdr_seaice_conc', 'U10M', 'V10M']
    ratio = [1, 100, 1, 1]
    vars_delta = [False, True, False, False]
    data_anom = {}
    for i, var in enumerate(vars):
        print(var)
        # Read data and remove the seasonal cycle 
        data = tools.read_data(var, slicing=False, months='all', limit_year=False, compute=False) * ratio[i]
        data = data.sel(latitude=slice(30,90))
        data = data.differentiate('time', datetime_unit='D') if vars_delta[i] else data # difference first, and then remove seasonal cycle
        data_anom[var] = tools.remove_seasonal_cycle_and_detrend(data, detrend=True).compute()

    m=3; n=3; seed=1
    folder = '/home/pyfsiew/codes/som/node_dates/'
    #node_date = np.load(folder + 'node_dates_keeptrend_%sx%s_%s.npy'%(m,n,seed), allow_pickle='TRUE').item()
    node_date = np.load(folder + 'node_dates_%sx%s_%s.npy'%(m,n,seed), allow_pickle='TRUE').item()
    nodes = [*node_date]
    #H500_maps = [data_anom['H500_regrid_1x1'].sel(time=node_date[node]).mean(dim='time') for node in nodes] # This has a weird line at 180E/W
    H500_maps = [data_anom['H500'].sel(time=node_date[node]).mean(dim='time') for node in nodes]
    lag = 0 # No lag for difference
    node_date_lag = {node: pd.to_datetime(node_date[node]) + pd.Timedelta(days=lag) for node in nodes}
    ice_maps= [data_anom['cdr_seaice_conc'].sel(time=node_date_lag[node]).mean(dim='time', skipna=True) for node in nodes] # Should be nan values in some nodes
    U10m_maps = [data_anom['U10M'].sel(time=node_date_lag[node]).mean(dim='time') for node in nodes]
    V10m_maps = [data_anom['V10M'].sel(time=node_date_lag[node]).mean(dim='time') for node in nodes]
    quiver_grids = [(U10m_maps[i], V10m_maps[i]) for i in range(len(nodes))]
    #quiver_grids = [None for i in range(len(nodes))]
    ice_pvals = [stats.ttest_1samp(data_anom['cdr_seaice_conc'].sel(time=node_date_lag[node]), 0, axis=0, nan_policy='omit').pvalue for node in nodes]
    ice_pvals = [xr.DataArray(i, dims=ice_maps[0].dims, coords=ice_maps[0].coords) for i in ice_pvals]
    total_date = sum([len(node_date[node]) for node in nodes])
    freq = [len(node_date[node])/total_date*100 for node in nodes]
    # Average duration day
    # The self-transition probablity
    Pt = []
    all_dates = [i for node in nodes for i in node_date[node]]
    for node in nodes:
        date_lag0 = node_date[node]
        date_lag_plus1= [i+pd.Timedelta(days=1) for i in date_lag0]
        days_valid = np.in1d(date_lag_plus1, all_dates) # Some dates are at the end of Mar 31. Plus 1 equal to 1 Apr and we don't have data
        days_same_node = np.in1d(date_lag_plus1, date_lag0).sum()
        P = days_same_node.sum() / days_valid.sum()
        Pt.append(1/(1-P)) # follow Falkena1 et al. 2020
    reload(fig2); fig2.plotting(3,3, H500_maps, ice_maps, ice_pvals, quiver_grids, freq, Pt, region_boxes_bool=False)

def plotting(m,n, contour_grids, shading_grids, shading_pvals, quiver_grids, freq, avg_duration, region_boxes_bool=True):

    reload(tools)
    import matplotlib

    # Fill the last column of contour grid with another line
    #contour_grids = [tools.map_fill_white_gap(cg) for cg in contour_grids]

    row=m; col=n
    grid = row*col
    cmap= 'coolwarm'
    mapcolors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#eef7fa', '#ffffff', '#ffffff', '#fff6e5', '#fddbc7',  '#f4a582', '#d6604d', '#b2182b']
    mapcolors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#FFFFFF', '#FFFFFF', '#fddbc7', '#f4a582', '#d6604d', '#b2182b'][::-1]
    cmap= matplotlib.colors.ListedColormap(mapcolors)
    mapcolor_grid  = [cmap] * grid
    shading_level = np.linspace(-15,15,11) # For sea ice anomaly
    shading_level = np.linspace(-2,2,11) # For sea ice anomaly
    shading_level_grids = [shading_level] * grid
    contour_level= np.linspace(-200,200,11)
    contour_level_grids= [contour_level] * grid

    clabels_row = [''] * grid
    top_title = [''] * col
    left_title = [''] * row
    ind_titles = None
    leftcorner_text= ['C' + str(i+1) + ':' + str(round(f,1)) +'%' for i, f in enumerate(freq)]
    projection=ccrs.NorthPolarStereo()
    xsize=2.4; ysize=2.4
    if region_boxes_bool: # Add a box enclosing BKS region in the middle
        region_boxes = None
        lat1, lat2, lon1, lon2 = 64.5,84.5,10.5,99.5 # Following Siew et al. 2021
        region_boxes = [None]*4 + [tools.create_region_box(lat1, lat2, lon1, lon2)] + [None]*4
    else:
        region_boxes = [None] * m * n
    # For the pvals, mask the regions with white color
    #white_idx = np.where(shading_level==0)[0][0] + 1
    white_idx = np.where(shading_level==0)[0][0] + 1
    white_level = shading_level[white_idx]
    shading_pvals = [xr.where(((shading_grids[i]>-white_level) & (shading_grids[i]<white_level)), 9999, shading_pvals[i]) for i in range(len(shading_pvals))]

    #matplotlib.rcParams['hatch.linewidth'] = 1
    #matplotlib.rcParams['hatch.color'] = 'brown'
    #pval_hatches = [[[0, 0.05, 1000], ['////', None]]] * grid # Mask the significant regions
    pval_maps = None
    pval_hatches = None

    xylims = [((-3000000,3500000), (-4000000,800000))] * grid  # For regime (BKS)
    xylims = [((-3500000,4000000), (-4500000,900000))] * grid  # For regime (BKS)
    xlims = [-180,180]; ylims = (30,90)
    xlims = None; ylims = None

    plt.close()
    fig, ax_all = plt.subplots(row,col,figsize=(col*xsize, row*ysize), subplot_kw={'projection':projection})
    tools.map_grid_plotting(shading_grids, row, col, mapcolor_grid, shading_level_grids, clabels_row, top_titles=top_title,
             left_titles=left_title, ind_titles=ind_titles, projection=projection, xsize=xsize, ysize=ysize,
             gridline=False, region_boxes=region_boxes, ylim=ylims, xlim=xlims, set_xylim=xylims, fill_continent=True,
             colorbar=False, contour_map_grids=contour_grids, contour_clevels=contour_level_grids,
             leftcorner_text=leftcorner_text, box_outline_gray=True, pltf=fig, ax_all=ax_all.flatten(),
                                   pval_map=None, pval_hatches=None, coastlines=False, contour_lw=0.8)
    # Plot the pvalus as contours (H500 already use the contour space..sosad)
    for i, ax in enumerate(ax_all.flatten()):
        lons = shading_pvals[i].longitude.values
        lats = shading_pvals[i].latitude.values
        # To remove the 0 line in contour level
        ax.contour(lons, lats, shading_pvals[i], [0,0.5,1000], colors='green', linewidths=0.4, transform=ccrs.PlateCarree())
    # Plot the vectors
    scale = 20
    regrid_shape = 6
    for i, ax in enumerate(ax_all.flatten()):
        if quiver_grids[i] is None:
            continue
        lons = quiver_grids[i][0].longitude.values; lats=quiver_grids[i][1].latitude.values
        x_vector = quiver_grids[i][0]
        y_vector = quiver_grids[i][1]
        transform = ccrs.PlateCarree()
        Q = ax.quiver(lons, lats, x_vector.values, y_vector.values, headwidth=6, headlength=3,
                headaxislength=2, units='width', scale_units='inches', pivot='middle', color='black', 
                 width=0.01, scale=scale, transform=ccrs.PlateCarree(), regrid_shape=regrid_shape, zorder=2) # Regrid_shape is important. Larger means denser
    if not all(v is None for v in quiver_grids): # if one of element is not None
        qk = ax_all.flatten()[-1].quiverkey(Q, 1.15, 0.15, 5, "5 m/s", labelpos='S', labelsep=0.05, coordinates='axes') # For sic

    # Plot the right-corner text (Average duration day)
    rightcorner_text = [str(round(f,1)) +'D' for i, f in enumerate(avg_duration)]
    for i, ax in enumerate(ax_all.flatten()):
        t1 = ax.annotate(rightcorner_text[i], xy=(0.83, 0.98), xycoords='axes fraction', fontsize=10, verticalalignment='top', 
                                                        bbox=dict(facecolor='white', edgecolor='white', alpha=1, pad=0.001))
    # Setup the coloar bar
    cba = fig.add_axes([0.91, 0.3, 0.02, 0.4])
    cNorm  = matplotlib.colors.Normalize(vmin=shading_level[0], vmax=shading_level[-1])
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm,cmap=cmap)
    cticks = yticklabels = [round(i,1) for i in shading_level if i!=0]
    cb1 = matplotlib.colorbar.ColorbarBase(cba, cmap=cmap, norm=cNorm, orientation='vertical',
                ticks=cticks, extend='both')
    cb1.ax.set_yticklabels(cticks, fontsize=10)
    cb1.set_label('%/day', fontsize=10, rotation=0, y=-0.05, labelpad=-7)
    #cb1.ax.set_title('%/day', fontsize=8, y=-0.2, loc='right', pad=-5)
    fig_name = 'fig2_spatial_map'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=-0.53) # hspace is the vertical
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)
