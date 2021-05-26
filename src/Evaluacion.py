from sklearn.metrics import classification_report


class Evaluacion:

    def __init__(self, y_predicted, y_test):
        self.report = classification_report(y_test, y_predicted, output_dict=True, zero_division=1)

    def print_report(self, accuracy=False, precision=False, recall=False, f1=False):

        if accuracy:
            print("\t\t- Accuracy:",   round(self.report["accuracy"]                  * 100, 2), '%')
        if precision:
            print("\t\t- Precisón:",   round(self.report["weighted avg"]["precision"] * 100, 2), '%')
        if recall:
            print("\t\t- Recall:",     round(self.report["weighted avg"]["recall"]    * 100, 2), '%')
        if f1:
            print("\t\t- F1 score:", round(self.report["weighted avg"]["f1-score"]  * 100, 2), '%')

        print()
