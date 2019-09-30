import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ActivityRegularization
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler
from keras import regularizers
from keras import backend as K
import math

import random 
from math import cos,sin
from keras.preprocessing import image
from scipy import misc
from scipy.misc import imresize
from skimage import io
from os.path import join

img_rows, img_cols = 128, 128

def random_indices(batchsize, ratio):
    size = int(batchsize * ratio)
    return np.random.choice(batchsize, size, replace=False)

def flip(inputs, targets, batchsize, flip_size):
    """Flip image batch"""
    indices = random_indices(batchsize, flip_size)
#     indices = len(inputs)
#     flip_input = inputs[indices,:,::-1]
    for k in indices:
        inputs[k] = inputs[k,:,::-1]
        targets[k, ::2] = (1-targets[k, ::2])
        p_indices = [(0, 3), (1, 2), (4, 9), (5, 8), (6, 7), (11, 13)]
        for i, j in p_indices:
            targets[k,[2*i, 2*j]] = targets[k,[2*j, 2*i]]
            targets[k,[2*i+1, 2*j+1]] = targets[k,[2*j+1, 2*i+1]]


def rotate(inputs, targets, batchsize, rotate_size):
    indices = random_indices(batchsize, rotate_size)
#     indices = len(inputs)
    for k in indices:
        angle = np.random.randint(-10, 10)
        inputs[k] = misc.imrotate(inputs[k, :, :, :], angle)
        inputs[k] = inputs[k].astype('float32')/255.
        angle = np.radians(angle)
        matrix = np.array([[cos(angle), sin(angle)] ,[-sin(angle), cos(angle)]])
        temp =  np.array(targets[k])
#         print (temp_coords.shape)
        for i in range(0,14):
            temp[2*i: 2*i+2] = temp[2*i: 2*i+2] - 0.5
            temp[2*i: 2*i+2] = matrix.dot(temp[2*i: 2*i+2])
            temp[2*i: 2*i+2] = temp[2*i: 2*i+2] + 0.5
        targets[k] = temp


def crop(inputs, targets, batchsize, crop_size):
    indices = random_indices(batchsize, crop_size)
#     indices = len(inputs)
    for k in indices:
        crop_x = np.random.randint(1, 10) 
        cro_y = np.random.randint(1, 10)
#         print(crop_x)
        imm = np.asarray(inputs[k, crop_y:, crop_x:,:])
        imm = imresize(imm, (img_rows, img_cols))
        imm = imm.astype('float32')/255.
        inputs[k] = imm
        targets[k,1::2] = (targets[k,1::2]*img_rows - crop_y)/(inputs[k, crop_y:, crop_x:]).shape[0]
        targets[k,::2] = (targets[k,::2]*img_cols - crop_x)/(inputs[k, crop_y:, crop_x:]).shape[1]
#     print(inputs.shape)


def yagenerator(folder, gt, batch_size):
    h, w = 128, 128
    names = list(gt.keys())
    names = names[:5000]
    num = len(names)
    #x = np.empty((batch_size, img_rows, img_cols, 3))
    #y = np.empty((batch_size, 28))
    
    while(True):
        random.shuffle(names)
        for i in range(num // batch_size - 1):
            x = np.empty((batch_size, img_rows, img_cols, 3))
            y = np.empty((batch_size, 28))
            
            for j in range(batch_size):
                ind = i * batch_size + j
                name = names[ind]
                #img = imread(join(folder, names[i]))
                img = image.load_img(join(folder, name))
                img = image.img_to_array(img)
                hr = 1 / img.shape[0]
                wr = 1 / img.shape[1]
                img = imresize(img, (h, w), interp='bilinear')
                x[j] = (img.astype('float32'))/255.
                
                y[j] = gt[name].copy()
                y[j, ::2] = y[j, ::2] * wr
                y[j, 1::2] = y[j, 1::2] * hr
            flip(x, y, batch_size, 0.7)
            rotate(x, y, batch_size, 0.6)
            crop(x, y, batch_size, 0.7)
            x -= aver
            x /= disp
#             x[(batch_size//4):2*(batch_size//4)], y[(batch_size//4):2*(batch_size//4)] = flip(
#                 x[:batch_size//4], y[:batch_size//4], batch_size//4)
#             x[2*(batch_size//4):3*(batch_size//4)], y[2*(batch_size//4):3*(batch_size//4)] = rotate(
#                 x[:batch_size//4], y[:batch_size//4], batch_size//4)
#             x[3*(batch_size//4):], y[3*(batch_size//4):] = crop(
#                 x[:batch_size//4], y[:batch_size//4], batch_size//4)
#             for n in range(0,3):
#                     mean_val = np.mean(x[:,:,:,n])
#                     std_val = np.std(x[:,:,:,n])
#                     x[:,:,:,n] = (x[:,:,:,n] - mean_val)/(std_val + 1e-8)
            yield (x, y)




def train_detector(train_gt, train_img_dir, fast_train=True):
    model = Sequential()
    # first set of CONV => RELU => POOL
    model.add(Conv2D(16, (3, 3), padding="same",
        input_shape=(128, 128, 3), data_format='channels_last'))#, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation("relu"))
    # model.add(ActivityRegularization(l2=0.01))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    # model.add(Dropout(0.1))
    # model.add(BatchNormalization())
    # second set of CONV => RELU => POOL
    model.add(Conv2D(32, (3, 3), padding="same"))#, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation("relu"))
    # model.add(ActivityRegularization(l2=0.01))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding="same"))#, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation("relu"))
    # model.add(ActivityRegularization(l2=0.01))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    # model.add(Dropout(0.2))
    model.add(Conv2D(100, (3, 3), padding="same"))#, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation("relu"))
    # model.add(ActivityRegularization(l2=0.01))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    # set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(160))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(28))
    model.add(Activation("sigmoid"))

    print("> Compiling...")
    model.compile(loss = 'mean_squared_error', optimizer='adam')
    
    if fast_train == True:
        model.fit_generator(yagenerator(train_img_dir, train_gt, batch_size=32), steps_per_epoch=4, epochs=10, 
                            verbose=1)
    else: 
        model.fit_generator(yagenerator(train_img_dir, train_gt, batch_size=64), steps_per_epoch=50000//64, epochs=10, 
                    verbose=1, validation_data=(X_test, Y_test))
        
#     return model
        



