import os

from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix


class Evaluacion:

    def print_report(self, y_test, y_predicted, accuracy=False, precision=False, recall=False, f1=False):
        report = classification_report(y_test, y_predicted, output_dict=True, zero_division=1)
        if accuracy:
            print("- Accuracy:",  round(report["accuracy"]                  * 100, 2), '%')
        if precision:
            print("- Precision:", round(report["weighted avg"]["precision"] * 100, 2), '%')
        if recall:
            print("- Recall:",    round(report["weighted avg"]["recall"]    * 100, 2), '%')
        if f1:
            print("- F1 score:",  round(report["weighted avg"]["f1-score"]  * 100, 2), '%')
        print()


    def plot_matrix(self, clf, X_test, y_test):
        plot_confusion_matrix(clf, X_test, y_test)


    def output(self, output_path, file_list, y_predicted):
        os.makedirs(output_path, exist_ok=True)
        output_file = open(output_path + "resultado.txt", "w+")

        for i in range(len(file_list)):
            output_file.write(str(file_list[i]) + "; " + str(int(y_predicted[i])).zfill(2) + '\n')
