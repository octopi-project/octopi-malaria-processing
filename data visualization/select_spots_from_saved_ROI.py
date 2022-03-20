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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data_id",nargs='?',help="input data id")
args = parser.parse_args()

if args.data_id != None:
    dataset_id = args.data_id
else:
    dataset_id = 'U3D_201910_2022-01-11_23-11-36.799392'

gcs_project = 'soe-octopi'
gcs_token = 'data-20220317-keys.json'

# dataset ID
bucket_destination = 'gs://octopi-malaria-data-processing'

# load spot data
fs = gcsfs.GCSFileSystem(project=gcs_project,token=gcs_token)
with fs.open( bucket_destination + '/' + dataset_id + '/' + 'spot_data_raw.csv', 'r' ) as f:
    spot_data_pd = pd.read_csv(f, index_col=None, header=0)

# initial gating
idx_spot_with_saturated_pixels = spot_data_pd['numSaturatedPixels']>0
spot_data_pd = spot_data_pd[~idx_spot_with_saturated_pixels]

# create scatter plot
R = spot_data_pd['R'].to_numpy()
G = spot_data_pd['G'].to_numpy()
B = spot_data_pd['B'].to_numpy()
s = np.vstack((R/B,G/B)).T
# FlowCal.plot.density2d(s, mode='scatter',xscale='linear',yscale='linear',xlim=[0,0.75],ylim=[0,1.5],xlabel='R/B',ylabel='G/B',title='',savefig="scatter plot.png")
# plt.show()

subplot_kw = dict(xlim=(0, 0.75), ylim=(0, 1.5), autoscale_on=False)
fig, ax = plt.subplots(subplot_kw=subplot_kw)
pts = ax.scatter(R/B,G/B,s=0.02,alpha=0.5)

plt.xlabel("R/B")
plt.ylabel("G/B")

# load ROI
ROI_path = pickle.load(open('path_ROI.p','rb'))

# select points
xys = pts.get_offsets()
Npts = len(xys)
fc = pts.get_facecolors()
fc = np.tile(fc, (Npts, 1))

idx_selected = np.nonzero(ROI_path.contains_points(xys))[0]

print(fc)

fc[:, 0] = 0.25
fc[:, 1] = 0.25
fc[:, 2] = 0.25
fc[idx_selected, -1] = 1
# set to red
fc[idx_selected, 0] = 0.75
fc[idx_selected, 1] = 0
fc[idx_selected, 2] = 0
pts.set_facecolors(fc)
ax.set_title('number of selected spots: ' + str(len(idx_selected)) + ' (out of ' + str(Npts) + ').')
ax.figure.canvas.draw_idle()

# idx_selected = idx_selected.tolist()
spot_data_selected_pd = spot_data_pd.iloc[idx_selected]
print(spot_data_selected_pd)
spot_data_selected_pd.to_csv('spot_data_selected_' + dataset_id + '.csv')

plt.savefig(dataset_id + '_selected spots.png')
plt.show()