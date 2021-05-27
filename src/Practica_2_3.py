import os

import numpy as np

from Aprendizaje import Aprendizaje
from Cargar_Datos import Cargar_Datos
from Comparador import Comparador
from Detector_MSER import Detector_MSER
from Entrenar import Entrenar
from Evaluar_Solucion import Evaluar_Solucion
from Limpieza import Limpieza
from Reconocimiento import Reconocimiento
from Representar import Representar
from time import time


class Practica_2_3:
    def __init__(self, detector, train_path, test_path):
        self.dir_train = train_path
        self.dir_test = test_path
        self.output_path = "resultado/resultado_imgs/"
        self.vers = 0

        self.aprendizaje = Aprendizaje()
        self.reconocimiento = Reconocimiento()
        self.load = Cargar_Datos()

        self.start = time()

        if (detector == 'Básico') or (detector == 'Basico'):
            self.vers = 0
            self.dimensiones = (25, 25)
        elif detector == 'Mejorado':
            self.vers = 1
            self.dimensiones = (30, 30)
        elif detector == 'Hough':
            self.vers = 2
            self.dimensiones = (30, 30)
        else:
            raise Exception("Utiliza una de las siguientes versiones: Básico, Mejorado, Hough")

        self.ejecutar()


    def ejecutar(self):
        print("Entrenando detector...")
        entrenador = Entrenar(vers=self.vers, directory=self.dir_train, dimensiones=self.dimensiones)
        clases_senal = entrenador.get_clases()  # clases_senal es un array de objetos clase_senal
        print("Listo.\n")

        print("Extrayendo detecciones...", end="")
        detector_mser = Detector_MSER(vers=self.vers, dimensiones=self.dimensiones)
        evaluador = Evaluar_Solucion()
        for file in os.listdir(self.dir_test):
            path = self.dir_test + file
            if file.split('.')[1] != 'txt':
                detector_mser.detectar(path)
            else:
                evaluador.set_file(path)

        print("\nExtracción completada.\n")
        detecciones = detector_mser.detecciones

        print("Puntuando detecciones...")
        comparador = Comparador(vers=self.vers, clases_senal=clases_senal, dimensiones=self.dimensiones)
        scored_detecciones, X_train_no, y_train_no, X_test_si = comparador.score_y_clasificar_no_senales(detecciones, descriptor_type='hog')
        print("\nListo.\n")

##      ## <Clasificador 1>
        X_train, y_train = self.load.load_senales(self.dir_train, X_train_no=X_train_no, y_train_no=y_train_no, descriptor_type='hog', dimensiones=(30, 30))
        lda = self.aprendizaje.entrenar_LDA(X_train, y_train)

        y_predicted = self.reconocimiento.clasificar_LDA(lda, X_test_si)

        # Quedarse con las buenas buenísimas
        index = y_predicted > 0
        scored_detecciones = np.array(scored_detecciones)
        scored_detecciones = scored_detecciones[index]
##      ## </Clasificador 1>

        print("Eliminando duplicados...")
        limpiador = Limpieza()
        detecciones_buenas = limpiador.limpiar(scored_detecciones)
        print("Listo.\n")

##      ## <Clasificador 2>
        X_train, y_train = self.load.load_data_train(self.dir_train, descriptor_type='hog', dimensiones=self.dimensiones)
        lda = self.aprendizaje.entrenar_LDA(X_train, y_train)

        for deteccion in detecciones_buenas:
            descriptor = deteccion.caracteristicas
            sub_type = self.reconocimiento.clasificar_LDA(lda, descriptor)
            deteccion.sub_type = sub_type
##      ## </Clasificador 2>

        print("Detecciones listas.")
        representador = Representar(path=self.output_path, color=(23, 23, 255))
        representador.representar(detecciones_buenas)

        print("Exportando imágenes...")
        output_file = open(self.output_path + "gt.txt", "w+")
        for i, deteccion in enumerate(detecciones_buenas):
            output_file.write(deteccion.to_string() + '\n')
            print(i, deteccion.to_string())
        print("Listo.\n")

        if evaluador.file_ok:
            evaluador.evaluar(detecciones_buenas)
            # ¿De las detecciones, cuántas son señales?:
            print("Recall:", evaluador.recall, "%")
            # ¿De las detecciones que había, cuántas se han detectado?:
            print("Precision:", evaluador.precision, "%")

        print("Duración:", round(time() - self.start, 4), "s")
