import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn import neighbors


class Aprendizaje:

    def entrenar_PCA(self, X, y):
        pca = PCA(len(np.unique(y)))
        pca.fit(X)

        return pca


    def reducir_PCA(self, pca, X):
        Z = pca.transform(X)
        # print("Reducción de la Dimensionalidad PCA:", X.shape[1], '-->', Z.shape[1])

        return Z


    def entrenar_LDA(self, X, y):
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)

        return lda


    def reducir_LDA(self, lda, X):
        Z = lda.transform(X)
        # print("Reducción de la Dimensionalidad LDA:", X.shape[1], '-->', Z.shape[1])

        return Z


    def entrenar_KNN(self, X, y, k=5):
        knn = neighbors.KNeighborsClassifier(k)
        knn.fit(X, y)

        return knn
