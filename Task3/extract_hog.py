import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import misc

def extract_hog(image):
    image = misc.imresize(image, (32, 32)).astype('float')
    if image.ndim > 2:
        i1y, i1x = [np.ascontiguousarray(g, dtype=np.double)
              for g in np.gradient(image[:,:,0])]
        i2y, i2x = [np.ascontiguousarray(g, dtype=np.double)
              for g in np.gradient(image[:,:,1])]
        i3y, i3x = [np.ascontiguousarray(g, dtype=np.double)
              for g in np.gradient(image[:,:,2])]
        num = np.argmax([np.sum(i1y**2 + i1x**2), np.sum(i2y**2 + i2x**2), np.sum(i3y**2 + i3x**2)])
        image = image[:,:, num]
#     ix = ndimage.filters.convolve(image, np.array([[-1, 0, 1]]))
#     iy = ndimage.filters.convolve(image, np.transpose([[-1, 0, 1]]))
    ix = ndimage.sobel(image, axis = 1)
    iy = ndimage.sobel(image, axis = 0)
#     iy, ix = [np.ascontiguousarray(g, dtype=np.double)
#               for g in np.gradient(image)]
    G = (ix**2 + iy**2)**0.5
    theta = (np.rad2deg(np.arctan2(iy, ix))) % 180
    n_bins = 9
    r_bins = 180 / n_bins
    H = np.zeros((4, 4, n_bins))
    for i in range(4):
        for j in range(4):
            magn = G[8*i : (8*i + 8), 8*j : (8*j + 8)]
            orient = theta[8*i : (8*i + 8), 8*j : (8*j + 8)]
    #         print(orient.shape)
            hist = np.zeros(n_bins)
            for y in range(orient.shape[0]):
                for x in range(orient.shape[1]):
#                     k = int(orient[y, x] // r_bins)
                    d, m = divmod(orient[y, x], r_bins)
                    if m == 0.0:
                        k = int(d-1)
                    else:
                        k = int(d)
                    hist[k] += magn[y, x]
#                     fract = (orient[y, x] - k*r_bins) / r_bins
#                         hist[1] += (1 - fract) * magn[y, x]
#                     else:
#                         hist[k] += fract * magn[y, x]
#                         if k == (n_bins - 1):
#                             hist[0] += (1 - fract) * magn[y, x]
#                         else:
#                             hist[k+1] += (1 - fract) * magn[y, x]
    #         print(cell)
    #         print(orient)
    #         print(hist)
            H[i, j, :] = hist/ 16.
    n_block = 2
    eps = 0.00001
    V = np.array([])
    for i in range(0, 3, 1):
        for j in range(0, 3, 1):
    #         print(H[i, j, :].shape, H[i, j + 1, :].shape, H [i + 1, j, :].shape, H[i + 1, j + 1, :].shape)
            vec = np.concatenate((H[i, j, :], H[i, j + 1, :], H [i + 1, j, :], H[i + 1, j + 1, :]), axis=0)
            vec = vec / ((np.sum(vec**2) + eps)**0.5)
            V = np.append(V, vec)
#     print(V)
    return V