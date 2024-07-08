import numpy as np
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from script1 import SamplingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


datasets = [make_classification(n_samples = 5000, n_features = 4, n_informative = 2, weights = (5,1)), make_classification(n_samples = 5000, n_features = 4, n_informative = 2, weights = (99,1)), make_classification(n_samples = 5000, n_features = 4, n_informative = 2, weights = (9,1), flip_y = 0.05)]

method = [RandomOverSampler(random_state=0), ClusterCentroids(random_state=0), SMOTE(random_state=0)]

metrics = [f1_score, balanced_accuracy_score, precision_score, recall_score]

scores = np.zeros((len(datasets), len(method), len(metrics), 10))

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)

# dataset x model x metric x fold

for i in range(scores.shape[0]):
    X, y = datasets[i]
    for j in range(scores.shape[1]):
        met = method[j]
        for l, (train_index, test_index) in enumerate(rskf.split(X, y)):
            mod = SamplingClassifier(base_estimator = GaussianNB(), base_preprocessing = met)
            mod.fit(X[train_index], y[train_index])
            y_pred =  mod.predict(X[test_index])
            for k in range(scores.shape[2]):
                err = metrics[k]
                scores[i,j,k,l] = err(y[test_index],y_pred)
                print(err(y[test_index], y_pred))

print(scores)

np.save("scores.npy", scores)