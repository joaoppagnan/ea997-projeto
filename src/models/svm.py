import numpy as np
import sklearn
from sklearn.svm import SVC

class SVM_Model():
    def __init__(self, X_train:np.ndarray, y_train:np.ndarray, C:float=3.0):
        self.model = SVC(C=C, kernel="rbf")
        self.X_train = X_train
        self.y_train = y_train

    def fit(self):
        self.model.fit(self.X_train)
        pass

    def predict(self, X_test:np.ndarray):
        self.y_pred = self.model.predict(X_test)

    def eval_model(self, y_test:np.ndarray):
        f1  = sklearn.metrics.f1_score(self.y_test, self.y_pred, average='macro')
        acc = sklearn.metrics.accuracy_score(y_test, self.y_pred)
        return (float(f1), float(acc))