import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier


class kNN(BaseEstimator, ClassifierMixin):

    def fit(self, X, y ,k = 1):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.k_ = k
        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
    
        m = X.shape[0]
        n = self.X_.shape[0]
        d = X.shape[1]
        dists = np.zeros((m, n)) # distance matrix (m, n)

        for i in range(m):
            for j in range(n):
                val = 0
                for k in range(d):
                    val += (X[i][k] - self.X_[j][k]) ** 2
                dists[i][j] = np.sqrt(val)
        # Check if fit has been called
        check_is_fitted(self)
        # Choice of a label for each new X
        a = np.argsort(dists, axis = 1)[:,:self.k_]

        return mode(self.y_[a], axis = 1)[0].ravel()



# X, y = make_classification(n_samples = 500)

# X_train, X_test, y_train, y_test = train_test_split(X,y)

# mod = kNN()
# mod.fit(X_train, y_train)
# ypred = mod.predict(X_test)


# mod2 = KNeighborsClassifier(n_neighbors = 1)

# mod2.fit(X_train, y_train)
# ypred = mod2.predict(X_test)

# print(ypred - ypred)