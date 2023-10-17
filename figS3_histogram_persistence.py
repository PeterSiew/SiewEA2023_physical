import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from importlib import reload
from scipy import stats

import sys; sys.path.insert(0, '/home/pyfsiew/codes/')
import tools

if __name__ == "__main__":

    # Plot the bar chart of histogram of each day persistence

    #ax1.bar(x-0.15, low_plotting_bars[node], bar_width, label=node, bottom=bottom_low, color=node_col[node])
    #x1.bar(x+0.15, high_plotting_bars[node], bar_width, label=node, bottom=bottom_high, color=node_col[node])
    nodes = range(1,10)
    plt.close()
    fig, axs = plt.subplots(9,1, figsize=(4,7))
    for i, node in enumerate(nodes):
        lag_date, __, __  = tools.return_lag_date(node=node, max_day=18)
        days_reverse = list(lag_date.keys())[::-1]
        persist_days_total = [len(lag_date[i]) for i in days_reverse]
        persist_days_count = [persist_days_total[0]] + np.diff(persist_days_total).tolist()
        axs[i].bar(days_reverse, persist_days_count)
        #axs[i].legend(loc='upper right', frameon=False)
        axs[i].set_ylabel('Count')
        axs[i].set_xticks([])
        axs[i].set_ylim(0,220)
        axs[i].annotate('Cluster%s'%node, xy=(0.8, 0.75), xycoords='axes fraction', size=11)
        for loc in ['left', 'right', 'top']:
            axs[i].spines[loc].set_visible(False)
        #axs[i].tick_params(axis='x', which='both',length=0)
        #axs[i].tick_params(axis='y', which='both',length=2)
    axs[-1].set_xticks([1,5,10,15])
    axs[-1].set_xlabel('Persistent days')

        
    fig_name = 'persit_days_bar'
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300)
