import numpy as np

from Aprendizaje import Aprendizaje
from Evaluacion import Evaluacion
from Reconocimiento import Reconocimiento


## Main
train_dir = "data/train_recortadas"
test_dir = "data/test_reconocimiento"

aprendizaje = Aprendizaje()
reconocimiento = Reconocimiento()


# Ejercicio 1
print("-- Descriptores HOG --")

# Aprendizaje
aprendizaje.load_data(train_dir, descriptor_type='hog', dimensions=(30, 30))
lda = aprendizaje.entrenar_LDA()

# Reconocimiento
reconocimiento.load_data(test_dir, descriptor_type='hog', dimensions=(30, 30))
predicted_LDA = reconocimiento.clasificar_LDA(lda)

print("\tLDA Bayesiano")
Evaluacion(predicted_LDA, reconocimiento.y_test).print_report(accuracy=False, precision=True, recall=True, f1=True)


## Ejercicio 2.2: KNN
# Aprendizaje
Z_train_lda = aprendizaje.reducir_LDA(aprendizaje.X_train)
knn_lda = aprendizaje.entrenar_KNN(Z_train_lda, 5)
# Reconocimiento
Z_test_lda = aprendizaje.reducir_LDA(reconocimiento.X_test)
predicted_LDA_KNN = reconocimiento.clasificar_KNN(Z_test_lda, knn_lda)

print("\tKNN reducido con LDA")
Evaluacion(predicted_LDA_KNN, reconocimiento.y_test).print_report(accuracy=False, precision=True, recall=True, f1=True)


## Ejercicio 2.3: PCA
aprendizaje.entrenar_PCA()
# Aprendizaje
Z_train_pca = aprendizaje.reducir_PCA(aprendizaje.X_train)
knn_pca = aprendizaje.entrenar_KNN(Z_train_pca, 5)
# Reconocimiento
Z_test_pca = aprendizaje.reducir_PCA(reconocimiento.X_test)
predicted_PCA_KNN = reconocimiento.clasificar_KNN(Z_test_pca, knn_pca)

print("\tKNN reducido con PCA")
Evaluacion(predicted_PCA_KNN, reconocimiento.y_test).print_report(accuracy=False, precision=True, recall=True, f1=True)


## Ejercicio 2.3: LBP
print("-- Descriptores LBP --")

# Aprendizaje
aprendizaje.load_data(train_dir, descriptor_type='lbp', dimensions=(30, 30))
lda = aprendizaje.entrenar_LDA()

# Reconocimiento
reconocimiento.load_data(test_dir, descriptor_type='lbp', dimensions=(30, 30))
predicted_LDA = reconocimiento.clasificar_LDA(lda)

print("\tLDA Bayesiano")
Evaluacion(predicted_LDA, reconocimiento.y_test).print_report(accuracy=False, precision=True, recall=True, f1=True)

