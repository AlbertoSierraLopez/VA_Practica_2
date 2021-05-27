from Aprendizaje import Aprendizaje
from Cargar_Datos import Cargar_Datos
from Evaluacion import Evaluacion
from Reconocimiento import Reconocimiento


class Practica_2:

    def __init__(self, dir_train, dir_test, descriptor='hog', clasificador='lda', dimensiones=(30, 30)):
        self.dir_train = dir_train
        self.dir_test = dir_test
        self.descriptor = descriptor
        self.clasificador = clasificador
        self.dimensiones = dimensiones

        self.aprendizaje = Aprendizaje()
        self.reconocimiento = Reconocimiento()
        self.evaluacion = Evaluacion()
        self.load = Cargar_Datos()

        self.hog = self.load.get_hog()

        self.ejecutar()


    def ejecutar(self):
        X_train, y_train = self.load.load_data_train(self.dir_train, descriptor_type=self.descriptor, dimensiones=self.dimensiones)
        X_test, y_test = self.load.load_data_test(self.dir_test, descriptor_type=self.descriptor, dimensiones=self.dimensiones)


    # Ejercicio 1
        print("-- Descriptores HOG --")
    #   Aprendizaje
        lda = self.aprendizaje.entrenar_LDA(X_train, y_train)
    #   Reconocimiento
        predicted_LDA = self.reconocimiento.clasificar_LDA(lda, X_test, y_test)

        print("\tLDA Bayesiano")
        self.evaluacion.print_report(predicted_LDA, y_test, accuracy=False, precision=True, recall=True, f1=True)
        # evaluacion.plot_matrix(lda, reconocimiento.X_test, reconocimiento.y_test)
        # plt.show()


    # Ejercicio 2.2: KNN
    #   Aprendizaje
        Z_train_lda = self.aprendizaje.reducir_LDA(lda, X_train)
        knn_lda = self.aprendizaje.entrenar_KNN(Z_train_lda, y_train, 5)
    #   Reconocimiento
        Z_test_lda = self.aprendizaje.reducir_LDA(lda, X_test)
        predicted_LDA_KNN = self.reconocimiento.clasificar_KNN(knn_lda, Z_test_lda)

        print("\tKNN reducido con LDA")
        self.evaluacion.print_report(predicted_LDA_KNN, y_test, accuracy=False, precision=True, recall=True, f1=True)


    # Ejercicio 2.3: PCA
    #   Aprendizaje
        pca = self.aprendizaje.entrenar_PCA(X_train, y_train)
        Z_train_pca = self.aprendizaje.reducir_PCA(pca, X_train)
        knn_pca = self.aprendizaje.entrenar_KNN(Z_train_pca, y_train, 5)
    #   Reconocimiento
        Z_test_pca = self.aprendizaje.reducir_PCA(pca, X_test)
        predicted_PCA_KNN = self.reconocimiento.clasificar_KNN(knn_pca, Z_test_pca)

        print("\tKNN reducido con PCA")
        self.evaluacion.print_report(predicted_PCA_KNN, y_test, accuracy=False, precision=True, recall=True, f1=True)
