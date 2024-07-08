import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score


X, y = load_breast_cancer(return_X_y = True)


models = [GaussianNB(), KNeighborsClassifier(n_neighbors = 3), DecisionTreeClassifier()]

# Dataset x Models x KFolds

results = np.zeros((1,3,10))

#KFolds

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=3)




#Dataset
for i in range(results.shape[0]):

    #Models
    for j in range(results.shape[1]):
        mod = models[j]

        #Kfolds
        for k, (train_index, test_index) in enumerate(rskf.split(X, y)):
            pca = PCA(n_components=0.8).fit(X[train_index])
            X_pca = pca.transform(X)
            mod.fit(X_pca[train_index], y[train_index])
            results[i,j,k] = accuracy_score(y[test_index], mod.predict(X_pca[test_index]))



for i in range(3):
    print(f'For mod {i+1} with pca')
    print(np.round(np.mean(results[0,i]),3))
    print(np.round(np.std(results[0,i]),3))




X, y = load_breast_cancer(return_X_y = True)


models = [GaussianNB(), KNeighborsClassifier(n_neighbors = 3), DecisionTreeClassifier()]

# Dataset x Models x KFolds

results = np.zeros((1,3,10))

#KFolds

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=3)


#Dataset
for i in range(results.shape[0]):

    #Models
    
    for j in range(results.shape[1]):
        mod = models[j]

        #Kfolds

        for k, (train_index, test_index) in enumerate(rskf.split(X, y)):
            Kbest = SelectKBest(k = int(np.sqrt(X.shape[1]))).fit(X[train_index], y[train_index])
            X_Kbest = Kbest.transform(X)
            mod.fit(X_Kbest[train_index], y[train_index])
            results[i,j,k] = accuracy_score(y[test_index], mod.predict(X_Kbest[test_index]))



for i in range(3):
    print(f'For mod {i+1} with SelectKBest')
    print(np.round(np.mean(results[0,i]),3))
    print(np.round(np.std(results[0,i]),3))
