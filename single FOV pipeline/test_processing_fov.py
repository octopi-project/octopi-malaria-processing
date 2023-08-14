import os
import cv2
import time
import imageio
from utils import *
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from processing_pipeline import *

# setting
classification_th = 0.8

# model
model_path = 'model_perf_r34_b32.pt'
if torch.cuda.is_available():
    model = torch.load(model_path)
else:
    model = torch.load(model_path,map_location=torch.device('cpu'))
    print('<<< using cpu >>>')

# cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
dummy_input = torch.randn(1024, 4, 31, 31)  # Adjust as per your input shape
if torch.cuda.is_available():
    dummy_input = dummy_input.cuda()
_ = model(dummy_input)

# images
I_fluorescence = imageio.v2.imread('1_1_0_Fluorescence_405_nm_Ex.bmp')
I_fluorescence = I_fluorescence[:,:,::-1]
I_BF_left = imageio.v2.imread('1_1_0_BF_LED_matrix_left_half.bmp')
I_BF_right = imageio.v2.imread('1_1_0_BF_LED_matrix_right_half.bmp')

# process fov
I,score = process_fov(I_fluorescence,I_BF_left,I_BF_right,model,device,classification_th)
print(I.shape)

# export images
np.save('positives.npy',I*255)

# save prediction score
df = pd.DataFrame(score, columns=["output"])
df.to_csv('score.csv', index=True, index_label="index")

for i in range(len(I)):
    numpy2png(255*I[i],'result/'+str(i))