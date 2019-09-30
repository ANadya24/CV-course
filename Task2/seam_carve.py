import numpy as np
import skimage.io as io
import skimage
from matplotlib import pyplot as plt

def rgb_to_yuv(img):
#     im = np.zeros((img.shape[0], img.shape[1]))
    im = 0.299* img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    return im
    
def dx(img):
#     imx = np.roll(img, 1, axis=1)
    imx = img[:,2:]
#     print(imx.shape)
    imxx = img[:,0:-2]
#     print(imxx.shape)
#     imxx = np.roll(img, -1, axis=1)
    im_x = np.zeros(img.shape)
    im_x[:,1:-1] = imx-imxx
    im_x[:,0] = img[:,1] - img[:,0]
    im_x[:,-1] = img[:,-1] - img[:,-2] 
    return im_x
    
def dy(img):
    imy = img[2:, :]
    imyy = img[0:-2,:]
    im_y = np.zeros(img.shape)
    im_y[1:-1,:] = imyy-imy
    im_y[0,:] = img[1,:] - img[0,:]
    im_y[-1,:] = img[-1,:] - img[-2,:] 
    return im_y
    
def my_range(start, end, step):
    while start > end:
        yield start
        start += step
        
def add_vertical(im,image, idxs, mask):
    seam = np.zeros((image.shape[0],image.shape[1]+1))
    seam[np.arange(0,image.shape[0]), idxs] = 1
    imag = im.reshape(image.size)
    y = np.arange(0, image.shape[0])
    idxx = y*(image.shape[1]) + idxs
    imag = np.insert(imag, idxx+1, -1).reshape((image.shape[0],image.shape[1]+1))
    mask = np.insert(mask, idxx+1, 256*image.size).reshape((image.shape[0],image.shape[1]+1))
    idxx = np.where(imag == -1)
    for y,x in zip(idxx[0],idxx[1]):
        imag[y, x] = np.mean([imag[y, x-1 if (x-1)>0 else 0], 
                              imag[y, x+1 if (x+1)<image.shape[1] else image.shape[1]-1]])
    return imag, seam, mask
    
def remove_vertical(img, image, idxs, mask):
    seam = np.zeros(image.shape)
    seam[np.arange(0,image.shape[0]), idxs] = 1
    imag = img.reshape(image.size)
    y = np.arange(0, image.shape[0])
    idxx = y*image.shape[1] + idxs
    imag = np.delete(imag, idxx).reshape((image.shape[0],image.shape[1]-1))
    mask = np.delete(mask, idxx).reshape((image.shape[0],image.shape[1]-1))
    # imag = imag.transpose()
    return imag, seam, mask
    
def seam_carve(img, string, mask = None):
    im = rgb_to_yuv(img)
    if mask is None:
        mask = np.zeros(im.shape)  
    mask = mask * (256*im.size)
    imx = dx(im)
    imy = dy(im)
    imm = np.abs(np.sqrt(imx**2+imy**2))
    imm = imm + mask
    if (string == 'vertical shrink') or (string == 'vertical expand'):
        imm = imm.transpose()
        im = im.transpose()
    image = np.zeros(imm.shape)
    image[0,:] = imm[0,:]
    for y in range(1,imm.shape[0]):
        for x in range(0,imm.shape[1]):
            up = np.array([max(x-1,0), x, min(x+1, imm.shape[1]-1)])
            m = min(image[y-1,up[0]], image[y-1,up[1]], image[y-1,up[2]])
            image[y,x] = m + imm[y,x]
    idxs = np.zeros(imm.shape[0], dtype=int)
    idxs[imm.shape[0]-1] = np.argmin(image[image.shape[0]-1,:])
    for y in my_range(imm.shape[0]-1, 0, -1):
        x=idxs[y]
        down = np.array([x-1 if (x-1)>0 else 0, x, x+1 if (x+1)<imm.shape[1]-1 else imm.shape[1]-1])
        idxs[y-1] = down[np.argmin([image[y-1,down[0]], image[y-1,down[1]], image[y-1,down[2]]])]
    if string == 'horizontal shrink': 
        im, seam, mask = remove_vertical(im,image,idxs, mask)
    elif string == 'vertical shrink':
        im,seam, mask = remove_vertical(im,image,idxs, mask)
        im = im.transpose()
        seam = seam.transpose()
    elif string == 'horizontal expand':
        im, seam, mask = add_vertical(im,image,idxs, mask)
    else:
        im, seam, mask = add_vertical(im,image,idxs, mask)
        im = im.transpose()
        seam = seam.transpose()
        mask = mask.transpose()
    return im, mask, seam
        
    