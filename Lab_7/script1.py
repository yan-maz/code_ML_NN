import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import RandomOverSampler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

class SamplingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator = None, base_preprocessing = None):
        self.base_estimator = base_estimator
        self.base_preprocessing = base_preprocessing

    def fit(self, X, y):

        if self.base_preprocessing:
            ros = self.base_preprocessing
            X, y = ros.fit_resample(X, y)
        
        self.base_estimator.fit(X, y)
        return self

    def predict(self, X):
        
        return self.base_estimator.predict(X)



# X, y = make_classification(n_samples = 500)

# X_train, X_test, y_train, y_test = train_test_split(X,y)


# sclf = SamplingClassifier(base_estimator = GaussianNB(), base_preprocessing = RandomOverSampler(random_state = 0))
# sclf.fit(X_train,y_train)
# sclf.predict(X_test)

# print(sclf.predict(X_test))

# print(accuracy_score(y_test,sclf.predict(X_test)))