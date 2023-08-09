import numpy as np
import cv2
import pandas as pd

f = open('list_of_datasets.txt','r')
DATASET_ID = f.read()
DATASET_ID = DATASET_ID.split('\n')
f.close()

SAVE_LOCALLY = True

for dataset_id in DATASET_ID:

  # print('<processing ' + dataset_id + '>')
  # a = np.load(dataset_id[:-7]+'_boolmask_count.npy',allow_pickle=True)
  # print(a.item()['count'])

  columns = ['FOV_row','FOV_col','count']
  segmantation_stat_pd = pd.DataFrame(columns=columns)
  total_number_of_cells = 0

  for i in range(10):
    for j in range(10):
      
      # print( str(i) + '_' + str(j) )
      segmentation_result = np.load(dataset_id + '/0/' + str(i) + '_' + str(j) + '_f_BF_LED_matrix_dpc_seg.npy',allow_pickle=True)
      
      mask = segmentation_result.item()['masks']
      
      # count the number of cells
      number_of_cells = np.amax(mask)
      FOV_entry = pd.DataFrame.from_dict({'FOV_row':[i],'FOV_col':[j],'count':[number_of_cells]})
      segmantation_stat_pd = pd.concat([segmantation_stat_pd,FOV_entry])
      total_number_of_cells = total_number_of_cells + number_of_cells
      
      # save mask
      mask = mask > 0
      mask_uint8 = mask.astype('uint8')*255
      cv2.imwrite(dataset_id + '/0/' + str(i) + '_' + str(j) + '_mask.bmp',mask_uint8)
      # np.save(dataset_id + '/0/' + str(i) + '_' + str(j) + '_mask.npy',mask)

  # save stats
  segmantation_stat_pd.to_csv(dataset_id + '.csv')
  print(dataset_id + ': '  + str(total_number_of_cells))