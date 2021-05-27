import numpy as np


class Reconocimiento:


    def clasificar_LDA(self, lda, X, y_test=None):
        y_predicted = lda.predict(X)

        if y_test is not None:
            n_aciertos = np.sum(y_test == y_predicted)
            # print("Tasa de acierto:", round(n_aciertos / len(y_predicted) * 100, 2), '%')

        return y_predicted


    def clasificar_KNN(self, knn, X, y_test=None):
        y_predicted = knn.predict(X)

        if y_test is not None:
            n_aciertos = np.sum(y_test == y_predicted)
            # print("Tasa de acierto:", round(n_aciertos / len(y_predicted) * 100, 2), '%')
        return y_predicted


    def clasificar_BAYES(self, clf, X, y_test=None):
        _, y_predicted = clf.predict(np.float32(X))

        if y_test is not None:
            n_aciertos = np.sum(y_test == y_predicted)
            # print("Tasa de acierto:", round(n_aciertos / len(y_predicted) * 100, 2), '%')
        return y_predicted
