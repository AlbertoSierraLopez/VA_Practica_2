import os
import cv2 as cv
import numpy as np
import sys

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import neighbors
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern


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


def entrada_train(train_dir, type, hog=None, dimensions=(30, 30)):

    X = np.array([])
    y = np.array([])
    # Iterar sobre todas las sub-carpetas de train
    for sub_dir in os.listdir(train_dir):
        list_sub_dir = os.listdir(train_dir+"/"+sub_dir)
        X_sub = np.zeros((len(list_sub_dir), dimensions[0] * dimensions[1]))
        y_sub = np.ones((len(list_sub_dir), 1))
        # Iterar sobre todas las imágenes de una carpeta
        for i in range(len(list_sub_dir)):
            file = list_sub_dir[i]

            # Procesar imagen
            img = cv.imread(train_dir+"/"+sub_dir+"/"+file, 0)
            img = cv.equalizeHist(img)
            img = cv.resize(img, dimensions, interpolation=cv.INTER_LINEAR)

            descriptor = None
            if (type == 'hog') and (hog is not None):
                descriptor = hog.compute(img)
            elif type == 'lbp':
                radius = 3
                n_points = 8 * radius
                descriptor = local_binary_pattern(img, n_points, radius, method='uniform')

            if descriptor is None:
                raise Exception("No descriptor available")
            else:
                X_sub[i] = np.reshape(descriptor, (1, dimensions[0] * dimensions[1]))
                y_sub[i] = int(sub_dir) * y_sub[i]

        # Apilar arrays
        if (X.shape[0] < 1):
            X = X_sub
            y = y_sub
        else:
            X = np.vstack((X, X_sub))
            y = np.vstack((y, y_sub))

    return X, np.reshape(y, (y.shape[0], ))


def entrada_test(test_dir, type, hog=None, dimensions=(30, 30)):

    list_files = os.listdir(test_dir)
    list_files.pop(0)
    X = np.zeros((len(list_files), dimensions[0] * dimensions[1]))
    y = np.ones((len(list_files), 1))

    for i in range(len(list_files)):
        file = list_files[i]
        file_name = file.split('-')
        # Procesar imagen
        img = cv.imread(test_dir + "/" + file, 0)
        img = cv.equalizeHist(img)
        img = cv.resize(img, (30, 30), interpolation=cv.INTER_LINEAR)

        descriptor = None
        if (type == 'hog') and (hog is not None):
            descriptor = hog.compute(img)
        elif type == 'lbp':
            radius = 3
            n_points = 8 * radius
            descriptor = local_binary_pattern(img, n_points, radius, method='uniform')

        if descriptor is None:
            raise Exception("No descriptor available")
        else:
            X[i] = np.reshape(descriptor, (1, dimensions[0] * dimensions[1]))
            y[i] = int(file_name[0]) * y[i]

    return X, np.reshape(y, (y.shape[0], ))


def reducir_PCA(X_train, n_components):
    pca = PCA(n_components)
    pca.fit(X_train)

    Z = pca.transform(X_train)
    print("Reducción de la Dimensionalidad PCA:", X_train.shape[1], '-->', Z.shape[1])
    return Z, pca


def reducir_LDA(X_train, y_train):
    lda = LinearDiscriminantAnalysis()

    lda.fit(X_train, y_train)

    Z = lda.transform(X_train)
    print("Reducción de la Dimensionalidad LDA:", X_train.shape[1], '-->', Z.shape[1])

    return Z, lda

def clasificar_LDA(X_train, y_train, X_test, y_test):
    Z, lda = reducir_LDA(X_train, y_train)

    y_predicted = lda.predict(X_test)
    n_aciertos = np.sum(y_predicted == y_test)
    # print("Tasa de acierto:", round(n_aciertos / len(y_predicted) * 100, 2), '%')
    return Z, y_predicted

def clasificar_KNN(X_train, y_train, X_test, y_test, k=5):
    knn = neighbors.KNeighborsClassifier(k).fit(X_train, y_train)

    y_predicted = knn.predict(X_test)
    n_aciertos = np.sum(y_test == y_predicted)
    return y_predicted


def report(y_test, y_predicted):
    report = classification_report(y_test, y_predicted, output_dict=True, zero_division=1)
    print("\tAccuracy:", round(report["accuracy"] * 100, 2), '%')
    # print("\tPrecisón:", round(report["weighted avg"]["precision"] * 100, 2), '%')
    # print("\tRecall:", round(report["weighted avg"]["recall"] * 100, 2), '%')
    print("\tF1 - score:", round(report["weighted avg"]["f1-score"] * 100, 2), '%')
    # print("Weighted avg:", report["macro avg"])
    print()


## Main
np.set_printoptions(threshold=sys.maxsize)
hog = get_hog()
train_dir = "data/train_recortadas"
test_dir = "data/test_reconocimiento"

## Ejercicio 2.1
# X_train, y_train = entrada_train(train_dir, type='hog', dimensions=(30, 30))
# X_test, y_test = entrada_test(test_dir, type='hog', dimensions=(30, 30))


# Ejercicio 1
X_train, y_train = entrada_train(train_dir, type='hog', hog=hog, dimensions=(30, 30))
X_test, y_test = entrada_test(test_dir, type='hog', hog=hog, dimensions=(30, 30))

Z, predicted_LDA = clasificar_LDA(X_train, y_train, X_test, y_test)
print("LDA Bayesiano")
report(y_test, predicted_LDA)


## Ejercicio 2.2
# KNN
# Reducir LDA para knn
Z_train, lda = reducir_LDA(X_train, y_train)
Z_test = lda.transform(X_test)

predicted_KNN = clasificar_KNN(Z_train, y_train, Z_test, y_test, 3)
print("KNN")
report(y_test, predicted_KNN)

## Ejercicio 2.3
# PCA
# Reducir PCA para knn
Z_train, pca = reducir_PCA(X_train, len(np.unique(y_train)))
Z_test = pca.transform(X_test)

predicted_KNN = clasificar_KNN(Z_train, y_train, Z_test, y_test, 3)
print("KNN")
report(y_test, predicted_KNN)

