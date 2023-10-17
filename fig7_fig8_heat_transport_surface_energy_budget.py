import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
import datetime as dt
import xarray as xr
from scipy import stats
import ipdb

from importlib import reload
import sys; sys.path.insert(0, '/home/pyfsiew/codes/')
import tools
import fig78_composite_energy_transport as fig78

if __name__ == "__main__":

    ### Get all vaviables for plotting
    if False:# Figure 7
        vars = ['heat_advection_850hPa', 'DQVDT_DYN', 'T850', 'TQV', 'UFLXQV', 'VFLXQV']
        #var_ratio = [-24*3600, 2.26e6, 1, 1, 2.26e6, 2.26e6]
        var_ratio = [-24*3600, 24*3600, 1, 1, 24*3600, 24*3600] # This is the correct one
        shading_vars = ['heat_advection_850hPa', 'DQVDT_DYN', 'T850', 'TQV']
    elif True: # Figure 8
        vars = ['HFLUX', 'EFLUX', 'LWGAB']; var_ratio = [-1, -1, 1] # positive is downwards
        shading_vars = ['HFLUX', 'EFLUX', 'LWGAB']
    else: # Revision figures showing the comparison between MERRA2 and ERA5
        vars=['T2M_regrid_1x1', 'TQV', 'LWGAB']
        vars=['ERA5_T2M_daily_regrid_1x1','ERA5_column_moisture_daily_regrid_1x1','ERA5_surface_downward_longwave_daily_regrid_1x1']
        shading_vars = vars
        var_ratio = [1,1,1]

    vars_delta = [False] * len(vars)
    vars_data = {}
    for i, var in enumerate(vars):
        print(var)
        data = tools.read_data(var, months='all', slicing=False) * var_ratio[i] # it doesn't allow lead lag in this case
        data = data.sel(latitude=slice(50,90)).compute()
        data = data.differentiate('time', datetime_unit='D') if vars_delta[i] else data
        data = tools.remove_seasonal_cycle_and_detrend(data, detrend=True).compute()
        vars_data[var] = data

    ### Get the dates of different node or regime
    nodes = [1,2,3,4,5,6,7,8,9]; ABCtitles=['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I']
    nodes = [1,2]; ABCtitles=['A', 'B']
    folder = '/home/pyfsiew/codes/som/node_dates/'; m,n,seed = 3,3,1
    node_date= np.load(folder + 'node_dates_%sx%s_%s.npy'%(m,n,seed), allow_pickle='TRUE').item()
    ### Start to form the composite
    composites = {node:{var:None for var in vars} for node in nodes}
    pvals = {node:{var:None for var in vars} for node in nodes}
    for node in nodes:
        for var in vars:
            time_sel = pd.to_datetime(node_date[node]) + pd.Timedelta(days=0) 
            time_mask = np.in1d(time_sel, vars_data[var].time)
            time_sel = time_sel[time_mask]
            var3d_sel = vars_data[var].sel(time=time_sel) 
            composite_map = var3d_sel.mean(dim='time') 
            composites[node][var]=composite_map
            pval = stats.ttest_1samp(var3d_sel.values, 0, axis=0, nan_policy='omit').pvalue
            pval = xr.DataArray(pval, dims=composite_map.dims, coords=composite_map.coords)
            pvals[node][var]=pval
    ### Do the plotting
    if False: # Figure 7
        row, col = len(nodes), 4
        shading_grids = [composites[node][var] for node in nodes for var in shading_vars]
        pvals_grids = [pvals[node][var] for node in nodes for var in shading_vars]
        quiver_grids = [[None,(composites[node]['UFLXQV'],composites[node]['VFLXQV']),None,None] for node in nodes]
        quiver_grids = [j for i in quiver_grids for j in i]
        #quiver_scales = [[None, 1e7, None, None] for node in nodes]; quiver_scales= [j for i in quiver_scales for j in i]
        quiver_scales = [[None, 3e7, None, None] for node in nodes]; quiver_scales= [j for i in quiver_scales for j in i]
        quiver_regrid_shape=[[None, 7, None, None] for node in nodes];quiver_regrid_shape=[j for i in quiver_regrid_shape for j in i]
        top_title = ['850 hPa temperature\nadvection', 'Column moisture\nconvergence', '850 hPa Temperature', 'Column moisture']
        xsize=2; ysize=2
    elif True: # Figure 8
        row, col = len(nodes), 3
        shading_grids = [composites[node][var] for node in nodes for var in shading_vars]
        pvals_grids = [pvals[node][var] for node in nodes for var in shading_vars]
        quiver_grids = [None for node in nodes for var in vars]
        quiver_scales=None
        quiver_regrid_shape=None
        top_title = ['Sensible heat flux', 'Latent heat flux', 'Downward longwave\nradiation']
        xsize=1.7; ysize=1.7
    elif True: # The reivision figure to comapre between MEERA2 and ERA5
        row, col = len(nodes), 3
        shading_grids = [composites[node][var] for node in nodes for var in shading_vars]
        pvals_grids = [pvals[node][var] for node in nodes for var in shading_vars]
        quiver_grids = [None for node in nodes for var in vars]
        quiver_scales=None
        quiver_regrid_shape=None
        top_title = ['T2M', 'Column moisture', 'Downward longwave\nradiation']
    else:
        pass
    #################
    regions = {'sic':((-1700000,3000000),(-3000000,600000)),
            'thickness':((-2500000,2100000),(-2600000,2600000)), 'surface_energy':((-1700000,2000000),(-2600000,1000))}
    map_types = ['sic'] * len(shading_grids)
    xylims =  [regions[m] for m in map_types]
    left_title = ['C%s'%node for node in nodes]
    cbarlabel = ''
    reload(fig78); fig78.plotting(row,col,shading_grids,pvals_grids,left_title, top_title,xylims=xylims, cbarlabel=cbarlabel,
        ABCtitles=ABCtitles,node=node,quiver_grids=quiver_grids,quiver_scales=quiver_scales,quiver_regrid_shape=quiver_regrid_shape, xsize=xsize, ysize=ysize)
    

def plotting(row, col, shading_grids, pvals_grids, left_title, top_title, xylims, cbarlabel='', ABCtitles='A', node=1, quiver_grids=None, quiver_scales=None, quiver_regrid_shape=None, xsize=2, ysize=2):

    reload(tools)
    map_grid = shading_grids
    grid = int(row*col)
    mapcolors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#FFFFFF', '#FFFFFF', '#fddbc7', '#f4a582', '#d6604d', '#b2182b'] 
    cmap= matplotlib.colors.ListedColormap(mapcolors)
    mapcolor = [cmap] * grid
    top_title = top_title
    left_title = left_title
    leftcorner_text = None
    xlims = [-130,130]
    xlims = [-180,180]
    ylims = (67,90)
    projection=ccrs.NorthPolarStereo()

    ### Setup the shading and clabel here
    if False: # Figure 7
        shading_levels = [np.linspace(-2.5,2.5,11),np.linspace(-2.5,2.5,11), np.linspace(-5,5,11),np.linspace(-2.5,2.5,11)] * row
        clabels_row=['$K day^{-1}$','$kg m^{-2} day^{-1}$', '$K$', '$kg m^{-2}$'] * row
        indiv_colorbar = [False]*col + [True]*col
    elif True: # Figure 8
        shading_levels = [np.linspace(-50,50,11)] * row *col
        clabels_row = ['$W m^{-2}$'] * row * col
        indiv_colorbar = [False]*col + [True]*col
    elif False: # revision figures
        shading_levels = [np.linspace(-4,4,11),np.linspace(-2.5,2.5,11),np.linspace(-30,30,11)] * row
        clabels_row = ['K','kgm-2','Wm-2'] * row
        indiv_colorbar = [False]*col + [True]*col
    else:
        pass

    region_boxes = None
    # Set the contour or hatches for significant
    contour_map_grids = None
    contour_clevels = None
    matplotlib.rcParams['hatch.linewidth'] = 1
    matplotlib.rcParams['hatch.color'] = 'lightgray'
    pval_map = pvals_grids
    pval_hatches = [[[0, 0.05, 1000], [None, 'XX']]] * grid # Mask the insignificant regions

    plt.close()
    fill_continent = False; coastlines=True # For energy flux
    fig, ax_all = plt.subplots(row,col,figsize=(col*xsize, row*ysize), subplot_kw={'projection':projection})
    ax_all = np.array(ax_all) if not isinstance(ax_all, list) else ax_all
    tools.map_grid_plotting(map_grid, row, col, mapcolor, shading_levels, clabels_row, top_titles=top_title, left_titles=left_title, projection=projection,
                        xsize=xsize, ysize=ysize, gridline=False, region_boxes=region_boxes, leftcorner_text=leftcorner_text, ylim=ylims, xlim=xlims,
                        pval_map=pval_map, pval_hatches=pval_hatches, set_xylim=xylims,
                    contour_map_grids=contour_map_grids, contour_clevels=contour_clevels, contour_color='green',
                    quiver_grids=None, shading_extend='both', fill_continent=fill_continent, colorbar=False, indiv_colorbar=indiv_colorbar,
                    pltf=fig, ax_all=ax_all.flatten(), coastlines=coastlines, box_outline_gray=True, transpose=False)

    ### Setup the quiver keys
    for i, ax in enumerate(ax_all.flatten()):
        if quiver_grids[i] is None:
            continue
        lons = quiver_grids[i][0].longitude.values; lats=quiver_grids[i][1].latitude.values
        x_vector = quiver_grids[i][0]
        y_vector = quiver_grids[i][1]
        transform = ccrs.PlateCarree()
        Q = ax.quiver(lons, lats, x_vector.values, y_vector.values, headwidth=6, headlength=3, headaxislength=2, units='width', scale_units='inches', 
                    pivot='middle',color='black',width=0.01, scale=quiver_scales[i], transform=ccrs.PlateCarree(), regrid_shape=quiver_regrid_shape[i], zorder=2) 
        #qk = ax.quiverkey(Q, 0.22, 0.9, 3e6, "$3x10^{6} kg m^{-1} day^{-1}$", labelpos='E', labelsep=0.05, coordinates='axes', fontproperties={'size': '10'})
        qk = ax.quiverkey(Q, 0.15, 0.9, 5e6, "$5x10^{6} kg m^{-1} day^{-1}$", labelpos='E', labelsep=0.05, coordinates='axes', fontproperties={'size': '10'})

    x=-0.13; y =1.4
    x=-0.17; y =0.9
    for i in range(row):
        ax_all[i,0].annotate(r'$\bf{(%s)}$'%ABCtitles[i], xy=(x, y), xycoords='axes fraction', size=11)

    ### Save the file
    fig_name = 'sea_ice_energy_flux'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=-0.33)
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)

