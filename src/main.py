import os
import cv2 as cv
import numpy as np

from sklearn.model_selection import train_test_split
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


def entrada_train(train_dir, hog):

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


def entrada_test(test_dir, hog):

    list_files = os.listdir(test_dir)
    list_files.pop(0)
    X = np.zeros((len(list_files), hog.getDescriptorSize())) #-1 porque la carpeta tiene un .directory
    y = np.ones((len(list_files), 1))

    for i in range(len(list_files)):
        file = list_files[i]
        file_name = file.split('-')
        # Procesar imagen
        img = cv.imread(test_dir + "/" + file, 0)
        img = cv.equalizeHist(img)
        img = cv.resize(img, (30, 30), interpolation=cv.INTER_LINEAR)

        descriptores = hog.compute(img)
        X[i] = np.reshape(descriptores, (1, hog.getDescriptorSize()))
        y[i] = int(file_name[0]) * y[i]

    return X, np.reshape(y, (y.shape[0], ))


## Main
hog = get_hog()
train_dir = "data/train_recortadas"
test_dir = "data/test_reconocimiento"

X, y = entrada_train(train_dir, hog)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

lda = LinearDiscriminantAnalysis()

lda.fit(X_train, y_train)
Z = lda.transform(X_train)
print("LDA:", X_train.shape, '-->', Z.shape)

y_predicted = lda.predict(X_test)
n_aciertos = np.sum(y_test == y_predicted)
print("Acierto sobre el mismo conjunto de datos:", round(n_aciertos/y_test.shape[0] * 100, 2), '%')


X_test_recon, y_test_recon = entrada_test(test_dir, hog)

y_predicted_recon = lda.predict(X_test_recon)
n_aciertos_recon = np.sum(y_test_recon == y_predicted_recon)
print("Acierto sobre el conjunto de datos de reconocimiento:", round(n_aciertos_recon/y_test_recon.shape[0] * 100, 2), '%')
