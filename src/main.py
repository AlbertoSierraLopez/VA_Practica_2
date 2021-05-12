import os
import cv2 as cv
import numpy as np

def entrada(train_dir):
    hog = cv.HOGDescriptor()
    X = np.empty((0, hog.getDescriptorSize()))
    y = np.empty((0, 1))
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

            winStride = (8, 8)
            padding = (8, 8)
            locations = ((10, 20),)
            descriptores = hog.compute(img, winStride, padding, locations)
            X_sub[i] = np.reshape(descriptores, (1, hog.getDescriptorSize()))

        # Apilar arrays
        #np.hstack(X, X_sub)
        #np.hstack(y, y_sub)


entrada("data/train_recortadas")