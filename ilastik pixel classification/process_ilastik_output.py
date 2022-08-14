import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import glob
import cv2
from utils_visualization import *

i = 0
j = 0

for i in range(26):
	for j in range(26):
		print(str(i) + '_' + str(j))
		I_overlay = cv2.imread(str(i) + '_' + str(j) + '.png')
		I = iio.imread(str(i) + '_' + str(j) + '_Probabilities.tif')
		I = I.astype('float')/np.iinfo(I.dtype).max # comment out this line on Ubuntu
		I = (I*255).astype('uint8')
		
		# hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
		# iio.imwrite('R.png',I[:,:,0])
		# iio.imwrite('G.png',I[:,:,1])
		# iio.imwrite('B.png',I[:,:,2])

		probability = I[:,:,0]
		probability = cv2.GaussianBlur(probability,(7,7),0)
		iio.imwrite('p_gaussian.png',probability)

		mask_binary = probability > 0.5*255
		mask_binary = np.uint8(mask_binary)*255
		iio.imwrite('mask.png',mask_binary*255)

		mask = I
		mask[:,:,0] = mask_binary
		mask[:,:,1] = mask_binary
		mask[:,:,2] = mask_binary

		numLabels,labels,stats,centroids = cv2.connectedComponentsWithStats(mask_binary,4,cv2.CV_32S)

		if numLabels > 1:
			for k in range(numLabels-1):
				area = stats[k+1, cv2.CC_STAT_AREA]
				x = round(centroids[k+1,0])
				y = round(centroids[k+1,1])
				if area > 50:
					print(area)
					save_cropped_spot(I_overlay,np.dstack((probability,probability,probability)),mask,i,j,x,y,'output')