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

    args = parser.parse_args()

    if args.classifier == "BAYES":
        clasificador = 'lda'
    elif args.classifier == "KNN":
        clasificador = 'knn'
    else:
        raise ValueError('Tipo de clasificador incorrecto')


    # Practica_2("data/train_recortadas/", "data/test_reconocimiento/", descriptor='hog', clasificador='lda', dimensiones=(30, 30))

    Practica_2(args.train_path, args.test_path, descriptor='hog', clasificador=clasificador, dimensiones=(30, 30))
