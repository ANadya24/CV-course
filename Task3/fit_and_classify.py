import numpy as np
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
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
        num = np.argmax([np.sum(i1y**2 + i1x**2),
                         np.sum(i2y**2 + i2x**2),
                         np.sum(i3y**2 + i3x**2)])
        image = image[:,:, num]
#     ix = ndimage.filters.convolve(image, np.array([[-1, 0, 1]]))
#     iy = ndimage.filters.convolve(image, np.transpose([[-1, 0, 1]]))
    ix = ndimage.sobel(image, axis = 1)
    iy = ndimage.sobel(image, axis = 0)
#     iy, ix = [np.ascontiguousarray(g, dtype=np.double)
#               for g in np.gradient(image)]
    G = (ix**2 + iy**2)**0.5
    theta = (np.rad2deg(np.arctan2(iy, ix))) % 360
    block_size = 4
    im_size = G.shape[0]
    n_bins = 20
    r_bins = 360 // n_bins
    H = np.zeros((im_size // block_size, im_size // block_size, n_bins))

    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            hist = H[i, j, :]
            magn = G[i*block_size : i*block_size + block_size, j*block_size : j*block_size + block_size]
            orient = theta[i*block_size : i*block_size + block_size, j*block_size : j*block_size + block_size]
            magn = magn.flatten()
            orient = orient.flatten()

            for (x, y) in zip(magn, orient):
                y = y % 360

                left = (y // r_bins).astype('int')
                right = left + 1
                fract_r = (y - left * r_bins) / r_bins
                fract_l = 1 - fract_r
                right = right % len(hist)

                hist[left] += x * fract_l
                hist[right] += x * fract_r

    eps = 0.001
    V = np.array([])
    for i in range(H.shape[0] - 1):
        for j in range(H.shape[1] - 1):
    #         print(H[i, j, :].shape, H[i, j + 1, :].shape, H [i + 1, j, :].shape, H[i + 1, j + 1, :].shape)
            vec = np.concatenate((H[i, j, :], H[i, j + 1, :], H [i + 1, j, :], H[i + 1, j + 1, :]))
            vec = vec / ((np.sum(vec**2) + eps)**0.5)
            V = np.append(V, vec)
#     print(V)
    return V

def fit_and_classify(train_features, train_labels, test_features):
    scaler = MinMaxScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    #clf = svm.SVC(kernel='rbf', C=10, gamma=0.25)#c=550
    clf = svm.SVC(kernel='linear', C=10)
    clf.fit(train_features, train_labels)
    return clf.predict(test_features)