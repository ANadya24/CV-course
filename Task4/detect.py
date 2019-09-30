import numpy as np
import keras
from keras.preprocessing import image
from scipy import misc
from scipy.misc import imresize
from skimage import io
from os.path import join
import os

def detect(model, test_img_dir):
    fnames = [name for name in os.listdir(test_img_dir) if name.endswith(".jpg")]
    points = dict.fromkeys(fnames)
    sizes = dict.fromkeys(fnames)
    batch_size = 64
    img_cols, img_rows = 128, 128
    
    for i in range(len(points) // batch_size + 1):
        x = np.empty((batch_size, img_rows, img_cols, 3))
        y = np.empty((batch_size, 28))

        for j in range(batch_size):
            ind = i * batch_size + j
            if ind >= len(points):
                break
            name = fnames[ind]
            #img = imread(join(folder, names[i]))
            img = image.load_img(join(test_img_dir, name))
            img = image.img_to_array(img)
            sizes[name] = img.shape
            hr = 1 / img.shape[0]
            wr = 1 / img.shape[1]
            img = imresize(img, (h, w), interp='bilinear')
            x[j] = (img.astype('float32'))/255.
        y = model.predict(x, batch_size, 0) 
        for j in range(batch_size):
            ind = i * batch_size + j
            name = fnames[ind]
            size = sizes[name]
            y[j,::2] *= size[1]
            y[j,1::2] *= size[0]
            points[name] = y[j]
    return points
    