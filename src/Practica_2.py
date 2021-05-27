import os
from matplotlib import pyplot as plt

from Aprendizaje import Aprendizaje
from Cargar_Datos import Cargar_Datos
from Evaluacion import Evaluacion
from Reconocimiento import Reconocimiento


class Practica_2:  # Nada, cuatro líneas de código :)

    def __init__(self, dir_train, dir_test, descriptor, clasificador, dimensionalidad, dimensiones=(30, 30)):
        self.dir_train = dir_train
        self.dir_test = dir_test
        self.descriptor = descriptor
        self.clasificador = clasificador
        self.dimensionalidad = dimensionalidad
        self.dimensiones = dimensiones

        self.aprendizaje = Aprendizaje()
        self.reconocimiento = Reconocimiento()
        self.evaluacion = Evaluacion()
        self.load = Cargar_Datos()

        self.hog = self.load.hog

        self.ejecutar()

    def ejecutar(self):

        X_train, y_train = self.load.load_data_train(self.dir_train, descriptor_type=self.descriptor,
                                                     dimensiones=self.dimensiones)
        X_test, y_test, file_list = self.load.load_data_test(self.dir_test, descriptor_type=self.descriptor,
                                                             dimensiones=self.dimensiones)

        if self.dimensionalidad == 'lda':
            lda = self.aprendizaje.entrenar_LDA(X_train, y_train)
            Z_train_lda = self.aprendizaje.reducir_LDA(lda, X_train)
            Z_test_lda = self.aprendizaje.reducir_LDA(lda, X_test)

            if self.clasificador == 'bayes':
                #   Reconocimiento
                y_predicted = self.reconocimiento.clasificar_LDA(lda, X_test, y_test)
                #   Evaluacion
                self.evaluacion.plot_matrix(lda, X_test, y_test)
                plt.show()

            elif self.clasificador == 'knn':
                #   Aprendizaje
                knn_lda = self.aprendizaje.entrenar_KNN(Z_train_lda, y_train, 5)
                #   Reconocimiento
                y_predicted = self.reconocimiento.clasificar_KNN(knn_lda, Z_test_lda)
                #   Evaluacion
                self.evaluacion.plot_matrix(knn_lda, Z_test_lda, y_test)
                plt.show()

            else:
                raise Exception("Clasificador inválido. Usa: BAYESIANO, KNN.")

        elif self.dimensionalidad == 'pca':
            pca = self.aprendizaje.entrenar_PCA(X_train, y_train)
            Z_train_pca = self.aprendizaje.reducir_PCA(pca, X_train)
            Z_test_pca = self.aprendizaje.reducir_PCA(pca, X_test)

            if self.clasificador == 'bayes':
                #   Aprendizaje
                bayes_pca = self.aprendizaje.entrenar_BAYES(Z_train_pca, y_train)
                #   Reconocimiento
                y_predicted = self.reconocimiento.clasificar_BAYES(bayes_pca, Z_test_pca)
            #   Evaluación
            # self.evaluacion.plot_matrix(bayes_pca, Z_test_pca, y_test)
            # plt.show()

            elif self.clasificador == 'knn':
                #   Aprendizaje
                knn_pca = self.aprendizaje.entrenar_KNN(Z_train_pca, y_train, 5)
                #   Reconocimiento
                y_predicted = self.reconocimiento.clasificar_KNN(knn_pca, Z_test_pca)
                #   Evaluación
                self.evaluacion.plot_matrix(knn_pca, Z_test_pca, y_test)
                plt.show()

        else:
            raise Exception("Algoritmo de reducción de la dimensionalidad inválido. Usa: LDA, PCA.")

        self.evaluacion.print_report(y_predicted, y_test, accuracy=False, precision=True, recall=True, f1=True)
        self.evaluacion.output("resultado/", file_list, y_predicted)
