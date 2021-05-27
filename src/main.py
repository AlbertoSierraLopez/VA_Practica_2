import argparse

from Practica_2 import Practica_2


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
        '--dimensionalidad', type=str, default="HOG", help='String con el nombre del algoritmo de dimensionalidad')

    args = parser.parse_args()

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

    # debug()


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

    Practica_2(args.train_path, args.test_path, descriptor=descriptor, clasificador=clasificador, dimensionalidad=dimensionalidad, dimensiones=(30, 30))
