import os
import cv2 as cv
import numpy as np

from skimage.feature import local_binary_pattern

class Cargar_Datos:

    def __init__(self):
        self.hog = self.get_hog()


    def get_hog(self):
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


    def load_data_train(self, train_dir, descriptor_type='hog', dimensiones=(30, 30)):
        X = np.array([])
        y = np.array([])
        # Iterar sobre todas las sub-carpetas de train
        for sub_dir in os.listdir(train_dir):
            list_sub_dir = os.listdir(train_dir + sub_dir)
            X_sub = np.zeros((len(list_sub_dir), dimensiones[0] * dimensiones[1]))
            y_sub = np.ones((len(list_sub_dir), 1))
            # Iterar sobre todas las imágenes de una carpeta
            for i in range(len(list_sub_dir)):
                file = list_sub_dir[i]

                # Procesar imagen
                img = cv.imread(train_dir + sub_dir + "/" + file, 0)
                img = cv.equalizeHist(img)
                img = cv.resize(img, dimensiones, interpolation=cv.INTER_LINEAR)

                descriptor = None
                if descriptor_type == 'hog':
                    descriptor = self.hog.compute(img)
                elif descriptor_type == 'lbp':
                    radius = 3
                    n_points = 8 * radius
                    descriptor = local_binary_pattern(img, n_points, radius, method='uniform')

                if descriptor is None:
                    raise Exception("No descriptor available")
                else:
                    X_sub[i] = np.reshape(descriptor, (1, dimensiones[0] * dimensiones[1]))
                    y_sub[i] = int(sub_dir) * y_sub[i]

            # Apilar arrays
            if X.shape[0] < 1:
                X = X_sub
                y = y_sub
            else:
                X = np.vstack((X, X_sub))
                y = np.vstack((y, y_sub))

        y = np.reshape(y, (y.shape[0],))

        return X, y


    def load_senales(self, train_dir, X_train_no, y_train_no, descriptor_type='hog', dimensiones=(30, 30)):
        X = np.array([])
        y = np.array([])
        # Iterar sobre todas las sub-carpetas de train
        for sub_dir in os.listdir(train_dir):
            list_sub_dir = os.listdir(train_dir + sub_dir)
            X_sub = np.zeros((len(list_sub_dir), dimensiones[0] * dimensiones[1]))
            y_sub = np.zeros((len(list_sub_dir), 1))
            # Iterar sobre todas las imágenes de una carpeta
            for i in range(len(list_sub_dir)):
                file = list_sub_dir[i]

                # Procesar imagen
                img = cv.imread(train_dir + sub_dir + "/" + file, 0)
                img = cv.equalizeHist(img)
                img = cv.resize(img, dimensiones, interpolation=cv.INTER_LINEAR)

                descriptor = None
                if descriptor_type == 'hog':
                    descriptor = self.hog.compute(img)
                elif descriptor_type == 'lbp':
                    radius = 3
                    n_points = 8 * radius
                    descriptor = local_binary_pattern(img, n_points, radius, method='uniform')

                if descriptor is None:
                    raise Exception("No descriptor available")
                else:
                    X_sub[i] = np.reshape(descriptor, (1, dimensiones[0] * dimensiones[1]))
                    y_sub[i] = 1

            # Apilar arrays
            if X.shape[0] < 1:
                X = X_sub
                y = y_sub
            else:
                X = np.vstack((X, X_sub))
                y = np.vstack((y, y_sub))

        X = np.vstack((X, X_train_no))
        y = np.concatenate([np.reshape(y, (y.shape[0], )), y_train_no])

        return X, y


    def load_data_test(self, test_dir, descriptor_type='hog', dimensiones=(30, 30)):
        print(test_dir)
        list_files = os.listdir(test_dir)
        # list_files.pop(0)
        X = np.zeros((len(list_files), dimensiones[0] * dimensiones[1]))
        y = np.ones((len(list_files), 1))

        file_list = []

        for i in range(len(list_files)):
            file = list_files[i]
            file_list.append(file)
            file_name = file.split('-')
            # Procesar imagen
            img = cv.imread(test_dir + file, 0)
            img = cv.equalizeHist(img)
            img = cv.resize(img, (30, 30), interpolation=cv.INTER_LINEAR)

            descriptor = None
            if descriptor_type == 'hog':
                descriptor = self.hog.compute(img)
            elif descriptor_type == 'lbp':
                radius = 3
                n_points = 8 * radius
                descriptor = local_binary_pattern(img, n_points, radius, method='uniform')

            if descriptor is None:
                raise Exception("No descriptor available")
            else:
                X[i] = np.reshape(descriptor, (1, dimensiones[0] * dimensiones[1]))
                y[i] = int(file_name[0]) * y[i]

        y = np.reshape(y, (y.shape[0],))

        return X, y, file_list
