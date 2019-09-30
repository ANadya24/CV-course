import keras
from keras.applications import ResNet50, InceptionV3, Xception
from keras import backend as K
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import  Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ActivityRegularization
from keras import regularizers
from keras import backend as K
import math

from math import cos,sin
from keras.preprocessing import image
from scipy import misc
from scipy.misc import imresize
from os.path import join
import os
import pickle

import matplotlib
import numpy as np
import random

def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            filename, class_id = line.rstrip('\n').split(',')
            res[filename] = int(class_id)
    return res

# def save_csv(img_classes, filename):
#     with open(filename, 'w') as fhandle:
#         print('filename,class_id', file=fhandle)
#         for filename in sorted(img_classes.keys()):
#             print('%s,%d' % (filename, img_classes[filename]), file=fhandle)
            
img_rows, img_cols = 224, 224
nb_classes = 50

aver = np.array([ 0.49004534 , 0.5139673  , 0.46879493])
disp = np.array([ 0.23112116 , 0.22527517 , 0.26772715])

def random_indices(batchsize, ratio):
    size = int(batchsize * ratio)
    return np.random.choice(batchsize, size, replace=False)

def flip(inputs, batchsize, flip_size):
    """Flip image batch"""
    indices = random_indices(batchsize, flip_size)
#     indices = len(inputs)
#     flip_input = inputs[indices,:,::-1]
    for k in indices:
        inputs[k] = inputs[k,:,::-1]


def rotate(inputs, batchsize, rotate_size):
    indices = random_indices(batchsize, rotate_size)
#     indices = len(inputs)
    for k in indices:
        angle = np.random.randint(-10, 10)
        inputs[k] = misc.imrotate(inputs[k, :, :, :], angle)
        inputs[k] = inputs[k].astype('float32')/255.


def crop(inputs, batchsize, crop_size):
    indices = random_indices(batchsize, crop_size)
#     indices = len(inputs)
    for k in indices:
        crop_x = np.random.randint(1, 10) 
        crop_y = np.random.randint(1, 10)
#         print(crop_x)
        imm = np.asarray(inputs[k, crop_y:, crop_x:,:])
        imm = imresize(imm, (img_rows, img_cols))
        imm = imm.astype('float32')/255.
        inputs[k] = imm
        

def yagenerator(folder, gt, batch_size, augment=False):
    h, w = 224, 224
    names = list(gt.keys())
#     names = names[:5000]
    num = len(names)
    #x = np.empty((batch_size, img_rows, img_cols, 3))
    #y = np.empty((batch_size, 28))
    
    while(True):
        random.shuffle(names)
        for i in range(num // batch_size - 1):
            x = np.empty((batch_size, img_rows, img_cols, 3))
            y = np.zeros((batch_size, nb_classes))
            
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
                y [j, gt[name]] = 1
#                 y[j] = gt[name]
            if augment == True:
                flip(x, batch_size, 0.7)
                rotate(x, batch_size, 0.6)
                crop(x, batch_size, 0.7)
            for col in range(0, 3):
                x[:,:,:,col] -= aver[col]
                x[:,:,:,col] /= disp[col]
            yield (x, y)

def train_classifier(train_gt, train_img_dir, fast_train=True):
    nb_train_samples = 2250
    nb_val_samples = 250
    nb_epoch = 20
    batch_size = 16
    img_height, img_width =  224, 224
    #train_data_dir = '/home/ec2-user/mynotebooks/train'
    #test_data_dir = '/home/ec2-user/mynotebooks/test'
    #train_gt = read_csv(join(train_data_dir, 'gt.csv'))
    #test_gt = read_csv(join(test_data_dir, 'gt.csv'))

    base_model = ResNet50(weights="imagenet", input_shape=(img_height, img_width, 3), include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x) #new FC layer, random init
    # predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
    # model = Model(input=base_model.input, output=predictions)
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(nb_classes, activation="softmax"))
    model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    print("compiled")
    if fast_train == True:
        hist = model.fit_generator(
        yagenerator(train_img_dir, train_gt, batch_size=batch_size, augment=False),
        nb_epoch=1,
        steps_per_epoch=1)
    else:
        hist = model.fit_generator(
        yagenerator(train_img_dir, train_gt, batch_size=batch_size, augment=True),
        nb_epoch=nb_epoch,
        steps_per_epoch=nb_train_samples//batch_size)
    return model

def classify(model, test_img_dir):
    fnames = [name for name in os.listdir(test_img_dir) if name.endswith(".jpg")]
    points = dict.fromkeys(fnames)
    
    batch_size = 16
    img_cols, img_rows = 224, 224
    
    for i in range(len(points) // batch_size + 1):
        x = np.empty((batch_size, img_rows, img_cols, 3))
        y = np.empty((batch_size, 50))

        for j in range(batch_size):
            ind = i * batch_size + j
            if ind >= len(points):
                break
            name = fnames[ind]
            #img = imread(join(folder, names[i]))
            img = image.load_img(join(test_img_dir, name))
            img = image.img_to_array(img)
            hr = 1 / img.shape[0]
            wr = 1 / img.shape[1]
            img = imresize(img, (img_cols, img_rows), interp='bilinear')
            x[j] = (img.astype('float32'))/255.
            
#             print(x.shape, aver.shape)
        for col in range(0,3):
            x[:,:,:,col] -= aver[col]
            x[:,:,:,col] /= disp[col]
        y = model.predict(x, batch_size, 0) 
        for j in range(batch_size):
            ind = i * batch_size + j
            if ind >= len(points):
                break
            name = fnames[ind]
            points[name] = np.argmax(y[j])
    return points
