import imageio
import numpy as np
import cv2

def make_movie(images_numpy, output, indices=None, scale_factor=5, fps=24, save_images=False):

    writer = imageio.get_writer(output + '.mp4', fps=fps)

    if indices is not None and save_images:
        if os.path.exists(output):
            os.rmdir(output)
        os.mkdir(output)

    for i in range(len(images_numpy)):
        frame = images_numpy[i]
        frame = frame.transpose(1,2,0)
        img_fluorescence = frame[:,:,[2,1,0]]
        img_dpc = frame[:,:,3]
        img_dpc = np.dstack([img_dpc,img_dpc,img_dpc])
        img_overlay = 0.64*img_fluorescence + 0.36*img_dpc
        frame = np.hstack([img_dpc,img_fluorescence,img_overlay]).astype('uint8')
        # print(frame.shape)
        # print(np.amax(frame))

        new_height, new_width = int(frame.shape[0] * scale_factor), int(frame.shape[1] * scale_factor)
        frame = cv2.resize(frame,(new_width, new_height),interpolation=cv2.INTER_NEAREST)
        writer.append_data(frame)

        if indices is not None and save_images: 
            imageio.imwrite(output + '/' + str(indices[i]) + '.png',frame)
        
        print(i)
        if i > 200:
            break

    writer.close()