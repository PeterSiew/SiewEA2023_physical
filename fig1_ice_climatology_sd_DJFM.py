import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import datetime as dt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
import ipdb

from importlib import reload
import sys; sys.path.insert(0, '/home/pyfsiew/codes/')
import tools

if __name__ == "__main__":

    # Plot the sea ice 15% sea ice edge and climatology in from NDJFM
    
    vars = ['cdr_seaice_conc', 'cice5_sic', 'piomas_aiday', 'ding22_aice']
    titles = ['Obs', 'CICE5', 'PIOMAS', 'Nudging']
    ratios = [100, 100, 100, 1]

    vars = ['cdr_seaice_conc_monthly', 'cice5_sic', 'NCAR_FOSI_aice']
    titles = ['OBS', 'CICE5', 'NCAR_FOSI']
    ratios = [100, 100, 100]

    vars = ['cice5_thickness', 'piomasAxel_hiday']
    titles = ['CICE5', 'PIOMAS']
    ratios = [1, 1] # in meter

    # Figure S1
    vars = ['cdr_seaice_conc', 'cice5_sic', 'piomas_aiday']
    titles = [r'$\bf{(A)}$' + ' NSIDC'] + [r'$\bf{(B)}$' + ' CICE5'] + [r'$\bf{(C)}$' + ' PIOMAS']
    ratios = [100, 100, 100]

    climatology, standard_deviation = [], []
    for i, var in enumerate(vars):
        ice_raw = tools.read_data(var, months='all', compute=True, limit_year=False) * ratios[i]
        ice_raw = ice_raw.sel(time=slice('1979-01-01', '2021-12-31'))
        ice_anom = tools.remove_seasonal_cycle_and_detrend(ice_raw, detrend=True)
        #ice_anom[:,:,:] = np.nan # Don't show the shading
        mon_mask = ice_raw.time.dt.month.isin([12,1,2,3])
        #mon_mask = ice_raw.time.dt.month.isin([6,7,8,9])
        ice_raw_sel = ice_raw.sel(time=mon_mask) 
        ice_anom_sel = ice_anom.sel(time=mon_mask) 
        climatology.append(ice_raw_sel.mean(dim='time'))
        standard_deviation.append(ice_anom_sel.std(dim='time'))

    # Plot the winter climatology
    col = len(vars)
    row = 1
    grid = row * col
    if True: # For Figure 1 (SIC climatology and standard deviation)
        contour_grids = climatology
        shading_grids = standard_deviation
        contour_levels = [[15]] * grid
        shading_levels = [range(5,45,5)] * grid
    else:
        # For thickness
        shading_grids = climatology
        contour_grids = None
        shading_levels = [np.arange(0.5,6,1)] * grid
        contour_levels = None
    clabels_row = [''] * grid
    left_title = [''] * row
    top_title = titles
    mapcolors = ['#ece7f2', '#d0d1e6', '#a6bddb', '#74a9cf', '#3690c0', '#0570b0', '#045a8d', '#023858']
    cmap= matplotlib.colors.ListedColormap(mapcolors)
    cmaps = [cmap] * grid
    title = "%s \n DJFM climatology" %var
    projection=ccrs.NorthPolarStereo(); xsize=2.5; ysize=2.5
    # Add a box enclosing BKS region in the middle
    lat1, lat2, lon1, lon2 = 64.5,84.5,10.5,99.5 # Following Siew et al. 2021
    region_boxes = [tools.create_region_box(lat1, lat2, lon1, lon2)] + [None]*(len(vars)-1)
    ylim = [60,90]
    xylims = [((-3500000,3200000), (-3500000,900000))] * grid  # For regime (BKS)
    xlim = None; ylim = None
    leftcorner_text = None
    ####
    plt.close()
    fig, ax_all = plt.subplots(row,col,figsize=(col*xsize, row*ysize), subplot_kw={'projection':projection})
    tools.map_grid_plotting(shading_grids, row, col, cmaps, shading_levels, clabels_row, top_titles=top_title,
                    left_titles=left_title,projection=projection, xsize=xsize, ysize=ysize, gridline=False,
                        region_boxes=region_boxes, leftcorner_text=leftcorner_text,contour_color='yellow',
                        ylim=ylim, xlim=xlim, shading_extend='max', fill_continent=True, coastlines=False,
                        contour_map_grids=contour_grids, contour_clevels=contour_levels, colorbar=True,
                        set_xylim=xylims, ax_all=ax_all, pltf=fig)
    ### Annotate the texts 
    ts = 7
    tc='orangered'
    transform = ccrs.PlateCarree()._as_mpl_transform(ax_all[1])
    ax_all[1].annotate('Greenland', xy=(-38, 67), xycoords=transform, ha='center', va='center', size=ts, color=tc)
    ax_all[1].annotate('Iceland', xy=(-17.6, 64.5), xycoords=transform, ha='center', va='center', size=ts, color=tc)
    ax_all[1].annotate('Scandinavia', xy=(26.1, 65.2), xycoords=transform, ha='center', va='center', size=ts, color=tc)
    ax_all[1].annotate('Barents\nSea', xy=(39.1, 72.5), xycoords=transform, ha='center', va='center', size=ts, color=tc)
    ax_all[1].annotate('Kara\nSea', xy=(74, 77), xycoords=transform, ha='center', va='center', size=ts, color=tc)
    ax_all[1].annotate('Taymyr\n      Peninsula', xy=(95, 72), xycoords=transform, ha='center', va='center', size=ts, color=tc)
    ax_all[1].annotate('Greenland\nSea', xy=(-8.2, 73.6), xycoords=transform, ha='center', va='center', size=ts, color=tc)
    ax_all[1].annotate('Urals', xy=(57.4, 64.4), xycoords=transform, ha='center', va='center', size=ts, color=tc)
    ax_all[1].annotate('Svalbard', xy=(17.1, 80), xycoords=transform, ha='center', va='center', size=ts, color=tc)
    ax_all[1].annotate('Labrador\nSea', xy=(-42, 56.1), xycoords=transform, ha='center', va='center', size=ts, color=tc)
    ###
    fig_name = 'sea_ice_concentration_and_sd'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)


