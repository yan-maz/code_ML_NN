import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from scipy.stats import mode


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classifiers = 1, soft = False, weighted = False):
        self.n_classifiers = n_classifiers
        self.classifiers = []
        self.soft = soft
        self.weighted = weighted
        for i in range(n_classifiers):
            self.classifiers.append(GaussianNB())

    def fit(self, X, y):

        self.accuracy = []

        n_samples = X.shape[0]
        pool_size = int(np.sqrt(n_samples))
        for i, clf in enumerate(self.classifiers):
            index = np.random.choice(n_samples, size = pool_size, replace = True)
            X_pool, y_pool = X[index], y[index]
            clf.fit(X_pool, y_pool)
            self.accuracy.append(accuracy_score(y[np.delete(np.arange(n_samples), index)], clf.predict(X[np.delete(np.arange(n_samples), index)])))

        for i in range(len(self.accuracy)):
            self.accuracy[i] = self.accuracy[i]/np.sum(self.accuracy)*self.n_classifiers




    def predict(self, X):
        if self.weighted:
            pred_list_proba_weighted = np.zeros((X.shape[0], self.n_classifiers))
            for i, clf in enumerate(self.classifiers):
                print(clf.predict_proba(X))
                pred_list_proba_weighted[:,i] = clf.predict_proba(X)[:,1] * self.accuracy[i]
            return(np.round(np.mean(pred_list_proba_weighted, axis = 1), decimals = 0))
        if self.soft:
            pred_list_proba = np.zeros((X.shape[0], self.n_classifiers))
            for i, clf in enumerate(self.classifiers):
                pred_list_proba[:,i] = clf.predict_proba(X)[:,1]
            return(np.round(np.mean(pred_list_proba, axis = 1), decimals = 0))
        else :  
            pred_list = np.zeros((X.shape[0], self.n_classifiers))
            for i, clf in enumerate(self.classifiers):
                pred_list[:,i] = clf.predict(X)
            return(mode(pred_list, axis = 1)[0])




from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

X, y = make_classification(n_samples=30)

models = [GaussianNB(), EnsembleClassifier(5, weighted = False), EnsembleClassifier(5, weighted = True)]

# Models x KFolds

results = np.zeros((3 ,10))

#KFolds

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=3)


#Models
for j in range(results.shape[0]):
    mod = models[j]

    #Kfolds
    for k, (train_index, test_index) in enumerate(rskf.split(X, y)):
        mod.fit(X[train_index], y[train_index])
        results[j,k] = accuracy_score(y[test_index], mod.predict(X[test_index]))
        
        
        
print(np.mean(results, axis = 1))