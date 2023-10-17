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
import fig6_composite_seaice as fig6


if __name__ == "__main__":


    ### Test the role of bottol melt from the CICE5 and PIOMAS (for response letter - but not used anymore)
    vars=['cice5_thickness_thermo', 'piomasAxel_iceprodday']
    vars=['cice5_thickness_thermo_bottom', 'cice5_thickness_thermo_top', 'cice5_thickness_thermo_lateral']
    vars=['cice5_thickness_thermo_bottom', 'piomasAxel_iceprodday_bottom']
    shading_vars=vars
    vars_delta = [False, False, False]
    var_ratio = [1,100,100]
    var_ratio = [1, -100*24*3600] # In PIOMAS here, melting is positive

    ### Figure 4 & Figure S4 (comparing obs total ice with the models)
    vars = ['cdr_seaice_conc', 'cice5_sic', 'piomas_aiday', 'cice5_thickness', 'piomasAxel_hiday']
    shading_vars = vars
    vars_delta = [True, True, True, True, True]
    var_ratio = [100, 100, 100, 100, 100]

    ### Figure S7 - decomposing sea ice conc and thickness into dynam and thermo component - for PIOMAS
    vars = ['cal_piomasAxel_sic_dynam', 'cal_piomasAxel_sic_thermo', 'piomasAxel_advectday',
            'piomasAxel_iceprodday', 'piomasAxel_udrift', 'piomasAxel_vdrift']
    shading_vars = vars[0:4]; vector_u_vars = vars[4]; vector_v_vars = vars[5]
    vars_delta = [False, False, False, False, False, False]
    var_ratio = [1, 1, 100*24*3600, 100*24*3600, 100, 100]

    ### Figure 8 Supp  (decomposing dynam thickness change into advection and convergence and their climatoogical term)
    vars = ['cal_cice5_thickness_dynam','cal_cice5_thickness_dynam_convergence','cal_cice5_thickness_dynam_advection', 
            'cal_cice5_thickness_dynam_advection_ubar_along_thicknessprime',
            'cal_cice5_thickness_dynam_advection_uprime_along_thicknessbar', 
            'cice5_icemove_u', 'cice5_icemove_v']
    shading_vars = vars[0:5]; vector_u_vars = vars[5]; vector_v_vars = vars[6]
    vars_delta = [False, False, False, False, False, False, False]
    var_ratio = [100, 100, 100, 1, 1, 100, 100]


    ### Figure 6 and Figure S6 - decomposing sea ice conc and thickness into dynam and thermo component - for CICE5
    vars=['cice5_sic_dynam','cice5_sic_thermo','cice5_thickness_dynam','cice5_thickness_thermo','cice5_icemove_u','cice5_icemove_v']
    shading_vars=vars[0:4]; vector_u_vars=vars[4]; vector_v_vars=vars[5]
    vars_delta = [False, False, False, False, False, False]
    var_ratio = [1, 1, 1, 1, 100, 100]

    vars_data = {}
    for i, var in enumerate(vars):
        print(var)
        data = tools.read_data(var, months='all', slicing=False) * var_ratio[i]
        data = data.sel(latitude=slice(60,90)).compute()
        data = data.differentiate('time', datetime_unit='D') if vars_delta[i] else data
        data = tools.remove_seasonal_cycle_and_detrend(data, detrend=True).compute()
        vars_data[var] = data

    ### Get the dates of different node or regime
    nodes = [8,9]; ABCtitles=['A', 'B']
    nodes = [1,2,3,4,5,6,7,8,9]; ABCtitles=['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I']
    nodes = [1,2]; ABCtitles=['A', 'B']
    folder = '/home/pyfsiew/codes/som/node_dates/'; m,n,seed = 3,3,1
    node_date= np.load(folder + 'node_dates_%sx%s_%s.npy'%(m,n,seed), allow_pickle='TRUE').item()
    #######################
    if False: # Limit the periods in the node_date (for the revision figures)
        node_date_new = {node:[] for node in nodes}
        for node in nodes:
            for day in node_date[node]:
                #if day.year<1991: # early period
                #if (day.year>=1991) & (day.year<=2000): 
                #if (day.year>2000) & (day.year<=2010): 
                if day.year>2010: # late period
                    node_date_new[node].append(day)
        node_date = node_date_new
    #############################
    ### Start to form the composite
    composites = {node:{var:None for var in vars} for node in nodes}
    pvals = {node:{var:None for var in vars} for node in nodes}
    for i, node in enumerate(nodes):
        for j, var in enumerate(vars):
            #time_sel = [pd.to_datetime(node_date[node])+pd.Timedelta(days=d) for d in [0,1]]
            #time_sel=pd.to_datetime([j for i in time_sel for j in i])# This gives the best result
            time_sel = pd.to_datetime(node_date[node]) + pd.Timedelta(days=0) 
            # Make sure all time is in the data. Regime dates from from reanalysis, 
            # while vars_data might be from models and satellite products (different covering dates?)
            time_mask = np.in1d(time_sel, vars_data[var].time)
            time_sel = time_sel[time_mask]
            ####
            var3d_sel = vars_data[var].sel(time=time_sel) 
            composite_map = var3d_sel.mean(dim='time') 
            composites[node][var]=composite_map
            pval = stats.ttest_1samp(var3d_sel.values, 0, axis=0, nan_policy='omit').pvalue
            pval = xr.DataArray(pval, dims=composite_map.dims, coords=composite_map.coords)
            pvals[node][var]=pval
    #############################################################33
    ### Do the plotting
    regions = {'sic':((-1700000,2300000),(-2600000,200000)), 'thickness':((-2500000,2100000),(-2600000,2600000))}
    if False: # Figure 4 main and S4
        row, col = 2, 5 
        top_title = ['NSIDC SIC', 'CICE5 SIC', 'PIOMAS SIC', 'CICE5 thickness', 'PIOMAS thickness']
        shading_grids = [composites[node][var] for node in nodes for var in shading_vars]
        pvals_grids = [pvals[node][var] for node in nodes for var in shading_vars]
        quiver_grids = [None for i in nodes for j in vars]
        quiver_scales=None; quiver_regrid_shape=None
        xsize=ysize=1.6
        map_types = ['sic']*len(shading_grids)
    elif True: # Fig 6, Fig S6 (CICE5), S7 (PIOMAS)
        row, col = len(nodes), 4 # Figure 8
        top_title = ['SIC\n(Dynamic)', 'SIC\n(Thermodyanmic)', 'Thickness\n(Dynamic)', 'Thickness\n(Thermodynamic)']
        shading_grids = [composites[node][var] for node in nodes for var in shading_vars]
        pvals_grids = [pvals[node][var] for node in nodes for var in shading_vars]
        quiver_grids = [[None,None,(composites[node][vector_u_vars],composites[node][vector_v_vars]),None] for node in nodes]
        quiver_grids = [j for i in quiver_grids for j in i]
        quiver_scales = [[None, None, 30, None] for node in nodes]; quiver_scales= [j for i in quiver_scales for j in i]
        quiver_regrid_shape = [[None, None, 8, None] for node in nodes]
        quiver_regrid_shape = [j for i in quiver_regrid_shape for j in i]
        xsize=ysize=2
        map_types = ['thickness']*len(shading_grids)
        map_types = ['sic']*len(shading_grids)
    elif False: # Figure S8:Decomposing the dynamic term into convergence and advection (decompose into climatology and anomaly)
        row, col = len(nodes), 5
        top_title = ['thickness', 'thicknes\nconvergence', 'thickness\nadvenction',
                    'thickness\nadvection\nubar*grad(Tprime)', 'thickness\nadvection\nuprime*grad(Tbar)']
        top_title = [r"$(-\nabla \cdot [uh])'$"] + [r"$(-h\nabla \cdot u)'$"] + [r"$(-u \nabla h)'$"] + \
                    [r"$-\overline{u} \nabla h'$"] + [r"$ -u' \nabla \overline{h}$"] 
        shading_grids = [composites[node][var] for node in nodes for var in shading_vars]
        pvals_grids = [pvals[node][var] for node in nodes for var in shading_vars]
        quiver_grids = [[(composites[node][vector_u_vars],composites[node][vector_v_vars]),None,None,None,None] for node in nodes]
        quiver_grids = [j for i in quiver_grids for j in i]
        quiver_scales = [[30, None, None, None, None] for node in nodes]; quiver_scales= [j for i in quiver_scales for j in i]
        quiver_regrid_shape=[[8,None,None,None,None] for node in nodes]
        quiver_regrid_shape=[j for i in quiver_regrid_shape for j in i]
        xsize=ysize=2
        map_types = ['sic']*len(shading_grids)
    elif False:
        row, col = len(nodes), len(vars)
        top_title = ['bottom', 'top', 'lateral']
        top_title = ['CICE5', 'PIOMAS']
        shading_grids = [composites[node][var] for node in nodes for var in shading_vars]
        pvals_grids = [pvals[node][var] for node in nodes for var in shading_vars]
        quiver_grids = [None for i in nodes for j in vars]
        quiver_scales=None; quiver_regrid_shape=None
        map_types = ['sic']*len(shading_grids)
    else:
        pass
    ######
    xylims =  [regions[m] for m in map_types]
    left_title = ['C%s'%node for node in nodes]
    cbarlabel = '%/day\ncm/day'
    reload(fig6); fig6.plotting(row, col, shading_grids, pvals_grids, left_title, top_title, xylims=xylims, cbarlabel=cbarlabel, 
        ABCtitles=ABCtitles,node=node,quiver_grids=quiver_grids,quiver_scales=quiver_scales,quiver_regrid_shape=quiver_regrid_shape,xsize=xsize,ysize=ysize)
    

def plotting(row, col, shading_grids, pvals_grids, left_title, top_title, xylims, cbarlabel='', ABCtitles='A', node=1, quiver_grids=None, quiver_scales=None, quiver_regrid_shape=None, xsize=2, ysize=2):

    reload(tools)
    shading_grids = [i+0.00001 for i in shading_grids]

    grid = int(row*col)
    mapcolors = ['#2166ac','#4393c3','#92c5de','#d1e5f0','#eef7fa','#ffffff','#ffffff','#fff6e5','#fddbc7','#f4a582','#d6604d','#b2182b'][::-1]
    cmap= matplotlib.colors.ListedColormap(mapcolors)
    mapcolor = [cmap] * grid
    clabels_row = [''] * grid
    left_title = left_title
    leftcorner_text = None
    #xlims = [-130,130]
    #xlims = [-180,180]
    #ylims = (67,90)

    lat1, lat2, lon1, lon2 = 64.5,84.5,10.5,99.5 # Following Siew et al. 2021
    region_boxes = [tools.create_region_box(lat1, lat2, lon1, lon2)] + [None]*(len(shading_grids)-1)
    region_boxes = None 

    projection=ccrs.NorthPolarStereo()
    shading_level = np.linspace(-2,2,11) # For sea ice 
    shading_level = np.linspace(-1,1,13) # For revision
    shading_level = np.linspace(-1.5,1.5,13) # For sea ice 
    shading_levels = [shading_level] * grid
    if False: # Set the contour as pvals
        contour_map_grids = pvals_grids
        contour_clevels = [[0, 0.05]] * grid
        white_idx = np.where(shading_level==0)[0][0] + 1
        white_level = shading_level[white_idx]
        # For the pvals, mask the regions with white color
        contour_map_grids = [xr.where(((shading_grids[i]>-white_level) & (shading_grids[i]<white_level)),
                    9999,contour_map_grids[i]) if shading_grids[i] is not None else None 
                            for i in range(len(contour_map_grids))]
    contour_map_grids = None
    contour_clevels = None
    matplotlib.rcParams['hatch.linewidth'] = 1
    matplotlib.rcParams['hatch.color'] = 'lightgray'
    pval_maps = pvals_grids
    pval_hatches = [[[0, 0.05, 1000], [None, 'XX']]] * grid # Mask the insignificant regions

    plt.close()
    fill_continent = True; coastlines=False # For sea ice 
    fig, ax_all = plt.subplots(row,col,figsize=(col*xsize, row*ysize), subplot_kw={'projection':projection})
    ax_all = np.array(ax_all) if not isinstance(ax_all, list) else ax_all
    tools.map_grid_plotting(shading_grids, row, col, mapcolor, shading_levels, clabels_row, top_titles=top_title, left_titles=left_title, projection=projection,
                    xsize=xsize, ysize=ysize, gridline=False, region_boxes=region_boxes, leftcorner_text=leftcorner_text, ylim=None, xlim=None,
                    pval_map=pval_maps, pval_hatches=pval_hatches, set_xylim=xylims,
                    contour_map_grids=contour_map_grids, contour_clevels=contour_clevels, contour_color='green',
                    quiver_grids=None, shading_extend='both', fill_continent=fill_continent, colorbar=False,
                    pltf=fig, ax_all=ax_all.flatten(), coastlines=coastlines, box_outline_gray=True)

    ### Setup the quiver keys
    for i, ax in enumerate(ax_all.flatten()):
        if quiver_grids[i] is None:
            continue
        lons = quiver_grids[i][0].longitude.values; lats=quiver_grids[i][1].latitude.values
        x_vector = quiver_grids[i][0]
        y_vector = quiver_grids[i][1]
        transform = ccrs.PlateCarree()
        Q = ax.quiver(lons, lats, x_vector.values, y_vector.values, headwidth=6, headlength=3, headaxislength=2, units='width', scale_units='inches', pivot='middle', color='black', 
                 width=0.01, scale=quiver_scales[i], transform=ccrs.PlateCarree(), regrid_shape=quiver_regrid_shape[i], zorder=2) # Regrid_shape is important. Larger means denser
        qk = ax.quiverkey(Q, 0.17, 0.9, 10, "10 cm/s", labelpos='S', labelsep=0.05, coordinates='axes') # For sic
    #if any([i is not None for i in quiver_grids]):
        #qk = ax.quiverkey(Q, 0.17, 0.9, 10, "10 cm/s", labelpos='S', labelsep=0.05, coordinates='axes') # For sic

    ### Set the (A), (B), (C) title
    x=-0.13; y =1.35 
    x=-0.22; y =0.83
    #ax_all.flatten()[0].annotate(r'$\bf{(%s)}$'%ABCtitle, xy=(x, y), xycoords='axes fraction', size=11)
    for i in range(row):
        ax_all[i,0].annotate(r'$\bf{(%s)}$'%ABCtitles[i], xy=(x, y), xycoords='axes fraction', size=11)

    if True: ### Setup the colorbar
        cba = fig.add_axes([0.2, 0.17, 0.6, 0.03]) # Figure 4 and 6 and S4
        #cba = fig.add_axes([0.2, 0.09, 0.6, 0.01]) # Figure S6,7
        cNorm  = matplotlib.colors.Normalize(vmin=shading_level[0], vmax=shading_level[-1])
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm,cmap=cmap)
        xticks = xticklabels = [round(i,2) for i in shading_level if i!=0]
        cb1 = matplotlib.colorbar.ColorbarBase(cba, cmap=cmap, norm=cNorm, orientation='horizontal',ticks=xticks, extend='both')
        cb1.ax.set_xticklabels(xticklabels, fontsize=10)
        cb1.set_label(cbarlabel, fontsize=10, x=1.12, labelpad=-22)

    if False: # Adjust the position of the subplots
        fig8.adjust_axis_position(ax_all.flatten(), row, col)
    if False: ### Save as pickle
        import pickle
        with open('./fig5%s_ice_map_cluster%s.pickle'%(ABCtitles, node), 'wb') as f: # should be 'wb' rather than 'w'
            pickle.dump(fig, f) 
        # To load the pickle
        #fig5 = pickle.load(open('fig8_seaice.pickle','rb'))

    ### Save the file
    fig_name = 'fig6_seaice_composition'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=-0.46) # Figure 4,6, S4
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.35, hspace=0) # Figure S6,7
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)

def adjust_axis_position(ax_all, rows, cols):

    gidx = []
    for i in range(rows):
        for j in range(cols):
            gidx.append((i,j))
    for i, ax in enumerate(ax_all):
        row=gidx[i][0]; col=gidx[i][1]
        if (row==0) & (col==0): # Skip the first map
            pass
        elif (row!=0) & (col==0): # Follow the map above (with same x and y)
            ax_p = ax_all[gidx.index((row-1,0))]
            x0_p, y0_p, x1_p, y1_p = ax_p.get_position().bounds
            x0 = x0_p
            x1 = x1_p
            y0 = y0_p - y1_p
            y1 = y1_p
            ax.set_position([x0, y0, x1, y1], which='both') # x-starting, y-starting, x-width, y-width
        else: # Follow the previous map (getting their end x-position)
            ax_p = ax_all[i-1]
            x0_p, y0_p, x1_p, y1_p = ax_p.get_position().bounds
            x0 = x0_p + x1_p
            x1 = x1_p 
            y0 = y0_p
            y1 = y1_p
            ax.set_position([x0, y0, x1, y1], which='both') 
            # if the set position doesn't match with what we assigned. This happens when there is a change of map size
            # Now we have the correct xwidth
            x1_n = ax.get_position().bounds[2] # Get the correct xwidth
            if x1_n != x1:
                ax.set_position([x0, y0, x1_n, y1], which='both') # To supply the correct xwidth. Others keep the same

