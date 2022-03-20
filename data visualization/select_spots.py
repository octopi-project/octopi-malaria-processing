import numpy as np
import pandas as pd
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import matplotlib.pyplot as plt
import FlowCal
import os
import pickle
from utils_gating import *
import gcsfs

gcs_project = 'soe-octopi'
gcs_token = 'data-20220317-keys.json'

# dataset ID
bucket_source = 'gs://octopi-malaria-tanzania-2021-data'
bucket_destination = 'gs://octopi-malaria-data-processing'
dataset_id = 'U3D_201910_2022-01-11_23-11-36.799392'

# load spot data
fs = gcsfs.GCSFileSystem(project=gcs_project,token=gcs_token)
with fs.open( bucket_destination + '/' + dataset_id + '/' + 'spot_data_raw.csv', 'r' ) as f:
    spot_data_pd = pd.read_csv(f, index_col=None, header=0)
# spot_data_pd = pd.read_csv('spot_data_raw.csv', index_col=None, header=0)

# initial gating
idx_spot_with_saturated_pixels = spot_data_pd['numSaturatedPixels']>0
spot_data_pd = spot_data_pd[~idx_spot_with_saturated_pixels]

# spot selection (modified from https://matplotlib.org/stable/gallery/widgets/lasso_selector_demo_sgskip.html)
R = spot_data_pd['R'].to_numpy()
G = spot_data_pd['G'].to_numpy()
B = spot_data_pd['B'].to_numpy()
s = np.vstack((R/B,G/B)).T
# FlowCal.plot.density2d(s, mode='scatter',xscale='linear',yscale='linear',xlim=[0,0.75],ylim=[0,1.5],xlabel='R/B',ylabel='G/B',title='',savefig="scatter plot.png")
# plt.show()

subplot_kw = dict(xlim=(0, 0.75), ylim=(0, 1.5), autoscale_on=False)
fig, ax = plt.subplots(subplot_kw=subplot_kw)
pts = ax.scatter(R/B,G/B,s=0.02,alpha=0.5)
# selector = SelectFromCollection(ax, pts, select_method='Lasso')
selector = SelectFromCollection(ax, pts, select_method='Polygon')

def accept(event):
    if event.key == "enter":
        print("Selected points:")
        idx = selector.ind.tolist()
        spot_data_selected_pd = spot_data_pd.iloc[idx]
        print(spot_data_selected_pd)
        spot_data_selected_pd.to_csv('spot_data_selected_' + dataset_id + '.csv')
        pickle.dump(selector.path,open('path_ROI.p','wb'))
        # selector.disconnect()
        ax.set_title('number of selected spots: ' + str(len(spot_data_selected_pd)) + '. Selection saved.')
        fig.canvas.draw_idle()        

fig.canvas.mpl_connect("key_press_event", accept)

plt.xlabel("R/B")
plt.ylabel("G/B")
plt.show()