import matplotlib.pyplot as plt
import matplotlib
import matplotlib; matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import xarray as xr
import ipdb
import sys
from datetime import date
from importlib import reload
import pandas as pd
sys.path.insert(0, '/home/pyfsiew/codes/')
import tools


if __name__ == "__main__":

    # Read the Obs sic
    # The first index is not really used here
    indices = ['BKSICE_raw', 'BKSICE_raw']; diffs=[False, True]; ratio=[100, 100]

    tss = []
    for i, index in enumerate(indices):
        ts = tools.read_timeseries(index, months='all', remove_winter_mean=False) * ratio[i]
        ts = ts.differentiate('time', datetime_unit='D') if diffs[i] else ts
        tss.append(ts)

    ice_raw = tss[0]
    ice_change = tss[1]
    # Get only DJFM data from ice change
    #mon_mask = ice_change.time.dt.month.isin([12,1,2,3])
    #ice_change = ice_change[mon_mask]

    #lag_day = 0
    #time_sel = ice_change.time + pd.Timedelta(days=lag_day)
    #ice_raw = ice_raw.sel(time=time_sel)

    ### Start the plot
    plt.close()
    fig, ax1 = plt.subplots(1,1,figsize=(4,4))
    #ax1.scatter(ice_raw, ice_change, s=0.5, color='k')
    corr = tools.correlation_nan(ice_raw, ice_change)
    print(corr)

    ax1.set_xlabel("Initial (raw) SIC concentration (0-100%)")
    ax1.set_ylabel('Daily SIC change (%/day)')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, zorder=1)
    ax1.axvline(x=0, color='gray', linestyle='--', linewidth=1, zorder=1)
    ax1.set_ylim(-3.5,3.5)
    ax1.set_xlim(20,85)

    # Get the node data 
    folder = '/home/pyfsiew/codes/som/node_dates/'
    m,n = 3,3; seed=1
    node_date = np.load(folder + 'node_dates_%sx%s_%s.npy'%(m,n,seed), allow_pickle='TRUE').item()
    nodes = [*node_date]
    node_col= {1:'#2166ac', 4:'#d1e5f0', 7:'#f4a582', 2:'#4393c3', 5:'#969696', 8:'#d6604d', 3:'#92c5de', 6:'#fddbc7', 9:'#b2182b'}
    for i, node in enumerate(nodes):
        ice_change_new = ice_change.sel(time=node_date[node])
        ice_raw_new = ice_raw.sel(time=node_date[node])
        ax1.scatter(ice_raw_new, ice_change_new, color=node_col[node], s=0.5)
        corr = tools.correlation_nan(ice_raw_new, ice_change_new)
        beta, intercept = tools.linregress_nan(ice_raw_new, ice_change_new)
        ax1.plot(ice_raw_new, ice_raw_new*beta+intercept, node_col[node])
        ax1.annotate(r"$C%s: \rho$ = %s" %(node, str(round(corr,2))), xy=(1.1, 0+i*0.1),
                xycoords='axes fraction', fontsize=9, color=node_col[node])
    fig_name = 'relationship_between_SIC_raw_and_change'
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(date.today(), fig_name), bbox_inches='tight', dpi=300)
