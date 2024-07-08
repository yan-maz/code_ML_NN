import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

class RandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self
    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Choice of a label for each new X
        y_pred = np.random.choice(self.classes_, X.shape[0])
        return y_pred




# X, y = make_classification(n_samples = 500)

# X_train, X_test, y_train, y_test = train_test_split(X,y)


# rdm = RandomClassifier()
# rdm.fit(X_train,y_train)
# rdm.predict(X_test)

# print(rdm.predict(X_test))