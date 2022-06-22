import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

class SVM_Model():
    def __init__(self, X_train:np.ndarray, y_train:np.ndarray, C:float=3.0):
        self.model = SVC(C=C, kernel="rbf")
        self.X_train = X_train
        self.y_train = y_train

    def fit(self):
        self.model.fit(self.X_train, self.y_train)
        pass

    def predict(self, X_test:np.ndarray):
        self.y_pred = self.model.predict(X_test)

    def eval_model(self, y_test:np.ndarray):
        f1  = sklearn.metrics.f1_score(y_test, self.y_pred, average='macro')
        acc = sklearn.metrics.accuracy_score(y_test, self.y_pred)
        return (float(f1), float(acc))

    def confusion_matrix(self, y_test:np.ndarray, fig_path:str):
        fig, ax = plt.subplots(tight_layout=True)
        ax = sns.heatmap(confusion_matrix(y_test, self.y_pred), annot=True, fmt=".0f", linewidths=.5)
        ax.set_ylabel("Classe verdadeira")
        ax.set_xlabel("Classe estimada")
        cfmatrix_name = fig_path + "svm_confusion_matrix.pdf"
        fig.savefig(cfmatrix_name)
        pass