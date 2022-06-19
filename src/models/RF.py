import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

class RF_Model():
    def __init__(self, X_train:np.ndarray, y_train:np.ndarray, n_estimators:int=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators)
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