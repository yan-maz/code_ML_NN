import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.base import clone
from strlearn.evaluators import TestThenTrain
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from scipy.stats import mode
from strlearn.streams import StreamGenerator
import matplotlib.pyplot as plt



class DSEnsemble_classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier = GaussianNB(), n_classifiers = 10):
        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.clf_pool = []


    def partial_fit(self, X, y, classes = 0):
        clf = clone(self.base_classifier)
        clf.fit(X,y)
        self.clf_pool.append(clf)
        if len(self.clf_pool)>self.n_classifiers:
            self.clf_pool.pop(0)
        return self


    def predict(self, X):
        results = []
        for clf in self.clf_pool:
            results.append(clf.predict(X))
        return(mode(results, axis = 0)[0][0])
        


clf = [GaussianNB(),DSEnsemble_classifier()]

stream = StreamGenerator(n_chunks=300, n_drifts=3, chunk_size = 200, weights=[0.95, 0.05], concept_sigmoid_spacing=999) #, incremental = False, n_features = 8, n_informative = 8, n_redundant = 0)

evaluator = TestThenTrain(metrics=(balanced_accuracy_score))

evaluator.process(stream, clf)



plt.plot(evaluator.scores[0], c = 'r')

plt.plot(evaluator.scores[1], c = 'b')

plt.savefig('fig')






