import numpy as np
import os
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import accuracy_score

datanames = os.listdir("datasets")
datanames.remove(".DS_Store")

# for data_id, data_name in enumerate(datanames):
#     print(data_name)
#     data = np.loadtxt("datasets/%s" % data_name, delimiter=',')
#     print(data)

models = [GaussianNB(), KNeighborsClassifier(n_neighbors = 3), DecisionTreeClassifier()]

# Dataset x Models x KFolds

results = np.zeros((len(datanames),len(models),10))

#KFolds

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=3)

#Dataset
for i in range(results.shape[0]):
    data = np.loadtxt("datasets/%s" % datanames[i], delimiter=',')
    X = data[:,:-1]
    y = data[:,-1]
    #Models
    for j in range(results.shape[1]):
        mod = models[j]

        #Kfolds
        for k, (train_index, test_index) in enumerate(rskf.split(X, y)):
            mod.fit(X[train_index], y[train_index])
            results[i,j,k] = accuracy_score(y[test_index], mod.predict(X[test_index]))


np.save("results.npy", results)