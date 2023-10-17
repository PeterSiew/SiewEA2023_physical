import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
import cartopy.crs as ccrs
import matplotlib
from importlib import reload
import pandas as pd
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"

import sys
sys.path.insert(0, '/home/pyfsiew/codes/')
import tools

if __name__ == "__main__":

    vars = ['T850', 'Q850']

    vars_data = {}
    for var in vars:
        data = tools.read_data(var, months='all', slicing=False, limit_lat=False)
        data = data.sel(latitude=slice(-30,90)).compute()
        vars_data[var] = data

    # Extract DJFM time
    mons = [12,1,2,3]
    idx = data.time.dt.month.isin(mons)
    map_grids = [vars_data[var].sel(time=idx).mean(dim='time') for var in vars]
    # plot the climatology
    row, col = len(vars), 1
    grid = row*col
    mapcolors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#FFFFFF', '#FFFFFF', '#fddbc7', '#f4a582', '#d6604d', '#b2182b'] 
    mapcolors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#FFFFFF'][::-1]
    cmap= matplotlib.colors.ListedColormap(mapcolors)
    mapcolor = [cmap] * grid
    mapcolor = ['Blues'] * grid
    shading_level_grid = [np.linspace(250,280,11), np.linspace(0.001,0.006,11)]
    top_title = [''] * col
    left_title = [''] * row
    ind_titles = ['(A) 850-hPa temperature', '(B) 850-hPa moisture']
    leftcorner_text=None
    projection = ccrs.PlateCarree(central_longitude=0); xsize=4; ysize=5
    xlims = [-180,180]
    xlims = [-60,120]
    ylims = (30,90)
    region_boxes = None
    contour_map_grids = None
    xylims = None
    indiv_colorbar = [True]*grid
    clabels_row= ['K', 'kg/kg']
    tools.map_grid_plotting(map_grids, row, col, mapcolor, shading_level_grid, clabels_row, top_titles=top_title,
            left_titles=left_title, ind_titles=ind_titles, projection=projection, xsize=xsize, ysize=ysize, gridline=False, region_boxes=region_boxes,
      ylim=ylims, xlim=xlims, set_xylim=xylims, fill_continent=False, colorbar=False, contour_map_grids=contour_map_grids, 
       contour_clevels=shading_level_grid, leftcorner_text=leftcorner_text,indiv_colorbar=indiv_colorbar,
       box_outline_gray=True)
    fig_name = 'T850_climatology'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=-0.7) # hspace is the vertical
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)
