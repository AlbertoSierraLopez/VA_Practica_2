import argparse
import os
import cv2 as cv

from Practica_2 import Practica_2
from Practica_2_3 import Practica_2_3


def debug():
    print("HOG, BAYES, LDA:")
    Practica_2("data/train_recortadas/", "data/test_reconocimiento/", descriptor='hog', clasificador='bayes', dimensionalidad='lda', dimensiones=(30, 30))
    print("HOG, KNN, LDA:")
    Practica_2("data/train_recortadas/", "data/test_reconocimiento/", descriptor='hog', clasificador='knn', dimensionalidad='lda', dimensiones=(30, 30))
    print("HOG, BAYES, PCA:")
    Practica_2("data/train_recortadas/", "data/test_reconocimiento/", descriptor='hog', clasificador='bayes', dimensionalidad='pca', dimensiones=(30, 30))
    print("HOG, KNN, PCA:")
    Practica_2("data/train_recortadas/", "data/test_reconocimiento/", descriptor='hog', clasificador='knn', dimensionalidad='pca', dimensiones=(30, 30))
    print("LBP, BAYES, LDA:")
    Practica_2("data/train_recortadas/", "data/test_reconocimiento/", descriptor='lbp', clasificador='bayes', dimensionalidad='lda', dimensiones=(30, 30))
    print("LBP, KNN, LDA:")
    Practica_2("data/train_recortadas/", "data/test_reconocimiento/", descriptor='lbp', clasificador='knn', dimensionalidad='lda', dimensiones=(30, 30))
    print("LBP, BAYES, PCA:")
    Practica_2("data/train_recortadas/", "data/test_reconocimiento/", descriptor='lbp', clasificador='bayes', dimensionalidad='pca', dimensiones=(30, 30))
    print("LBP, KNN, PCA:")
    Practica_2("data/train_recortadas/", "data/test_reconocimiento/", descriptor='lbp', clasificador='knn', dimensionalidad='pca', dimensiones=(30, 30))
    exit()


def separar_datos_entrada(test_path):
    dir1 = test_path + "practica2/"
    dir2 = test_path + "practica1/"
    os.makedirs(dir1, exist_ok=True)
    os.makedirs(dir2, exist_ok=True)
    for file in os.listdir(test_path):
        if not os.path.isdir(file):
            split_file = file.split('.')
            extension = split_file[len(split_file)-1]
            if extension == 'txt':
                os.rename(test_path + "/" + file, dir2 + "/" + file)
            elif extension == 'ppm' or extension == 'jpg':
                img = cv.imread(test_path + "/" + file)
                if (img.shape[0] > 150) and (img.shape[1] > 150):
                    os.rename(test_path + "/" + file, dir2 + "/" + file)
                else:
                    os.rename(test_path + "/" + file, dir1 + "/" + file)
    return dir1, dir2


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Entrena sober train y ejecuta el clasificador sobre imgs de test')
    parser.add_argument(
        '--train_path', type=str, default="./train", help='Path al directorio de imgs de train')
    parser.add_argument(
        '--test_path', type=str, default="./test", help='Path al directorio de imgs de test')
    parser.add_argument(
        '--classifier', type=str, default="BAYES", help='String con el nombre del clasificador')
    parser.add_argument(
        '--descriptor', type=str, default="HOG", help='String con el nombre del descriptor de imágenes')
    parser.add_argument(
        '--dimensionalidad', type=str, default="LDA", help='String con el nombre del algoritmo de dimensionalidad')

    args = parser.parse_args()

    # Debug
    debug()

    # Separar Datos de Entrada
    test_path1, test_path2 = separar_datos_entrada(args.test_path)

    # Comprobar Argumentos
    if args.classifier == "BAYES":
        clasificador = 'bayes'
    elif args.classifier == "KNN":
        clasificador = 'knn'
    else:
        raise ValueError('Tipo de clasificador incorrecto')

    if args.descriptor == "HOG":
        descriptor = 'hog'
    elif args.descriptor == "LBP":
        descriptor = 'lbp'
    else:
        raise ValueError('Tipo de descriptor incorrecto')

    if args.dimensionalidad == "LDA":
        dimensionalidad = 'lda'
    elif args.descriptor == "PCA":
        dimensionalidad = 'pca'
    else:
        raise ValueError('Tipo de algoritmo de reducción de la dimensionalidad incorrecto')

    if len(os.listdir(test_path1)) > 1:
        Practica_2(args.train_path, test_path1, descriptor=descriptor, clasificador=clasificador, dimensionalidad=dimensionalidad, dimensiones=(30, 30))
    if len(os.listdir(test_path2)) > 1:
        Practica_2_3('Mejorado', args.train_path, test_path2)
