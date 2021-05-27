from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix


class Evaluacion:

    def print_report(self, y_test, y_predicted, accuracy=False, precision=False, recall=False, f1=False):
        report = classification_report(y_test, y_predicted, output_dict=True, zero_division=1)

        if accuracy:
            print("\t\t- Accuracy:",   round(report["accuracy"]                  * 100, 2), '%')
        if precision:
            print("\t\t- Precision:",   round(report["weighted avg"]["precision"] * 100, 2), '%')
        if recall:
            print("\t\t- Recall:",     round(report["weighted avg"]["recall"]    * 100, 2), '%')
        if f1:
            print("\t\t- F1 score:", round(report["weighted avg"]["f1-score"]  * 100, 2), '%')
        print()


    def plot_matrix(self, clf, y_test, y_predicted):
        plot_confusion_matrix(clf, y_test, y_predicted)
