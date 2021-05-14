import os
import cv2 as cv
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_hog():
    winSize = (30, 30)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (5, 5)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    # gammaCorrection = 1
    gammaCorrection = False
    nlevels = 64
    signedGradient = True

    return cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)


def entrada(train_dir):

    hog = get_hog()

    X = np.array([])
    y = np.array([])
    # Iterar sobre todas las sub-carpetas de train
    for sub_dir in os.listdir(train_dir):
        list_sub_dir = os.listdir(train_dir+"/"+sub_dir)
        X_sub = np.zeros((len(list_sub_dir), hog.getDescriptorSize()))
        y_sub = np.ones((len(list_sub_dir), 1))
        # Iterar sobre todas las imágenes de una carpeta
        for i in range(len(list_sub_dir)):
            file = list_sub_dir[i]

            # Procesar imagen
            img = cv.imread(train_dir+"/"+sub_dir+"/"+file, 0)
            img = cv.equalizeHist(img)
            img = cv.resize(img, (30, 30), interpolation=cv.INTER_LINEAR)
            img = np.uint8(img)

            descriptores = hog.compute(img)
            X_sub[i] = np.reshape(descriptores, (1, hog.getDescriptorSize()))
            y_sub[i] = int(sub_dir) * y_sub[i]

        # Apilar arrays
        if (X.shape[0] < 1):
            X = X_sub
            y = y_sub
        else:
            X = np.vstack((X, X_sub))
            y = np.vstack((y, y_sub))

    return X, np.reshape(y, (y.shape[0], ))


def reducir_dimensionalidad(X, y):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    Z = lda.transform(X)
    return Z


X, y = entrada("data/train_recortadas")

Z = reducir_dimensionalidad(X, y)
