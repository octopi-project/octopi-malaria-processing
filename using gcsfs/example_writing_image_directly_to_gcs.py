import gcsfs
import zarr
import cv2

fs = gcsfs.GCSFileSystem(project='soe-octopi',token='whole-slide-20220214-keys.json')
with fs.open('gs://octopi-malaria-whole-slide/test.jpg','wb') as f:
	I = cv2.imread('test.png')
	I_str = cv2.imencode('.jpg',I)[1].tobytes()
	f.write(I_str)