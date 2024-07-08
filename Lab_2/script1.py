import numpy as np
from sklearn.datasets import make_classification 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from sklearn.model_selection import RepeatedStratifiedKFold

X, y  = make_classification(n_samples=400, n_informative=2, flip_y=0.08, random_state=1)

y = np.reshape(y, newshape = (400,1))

Dat = np.concatenate((X[:,0:2], y), axis=1)

np.savetxt('Data.csv', Dat)

fig = plt.scatter(X[:,0], X[:,1], c=y)

plt.savefig('scatter.png')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

clf = GaussianNB()


y_train = np.reshape(y_train, newshape = 320)
y_test = np.reshape(y_test, newshape = 80)

clf.fit(X_train, y_train)

matrix = clf.predict_proba(X_test)

y_pred = np.argmax(matrix, axis=1)


print(accuracy_score(y_test, y_pred))

fig, axs = plt.subplots(nrows=1, ncols=2)


# axs[0].plot(X_test[:,0], X_test[:,1], c=y_test)
# axs[1].plot(X_test[:,0], X_test[:,1], c=y_pred)

# plt.savefig('scatter2.png')


rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)

rskf.get_n_splits(X, y)
L = np.zeros(10)
for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    L[i] = accuracy_score(y_test, y_pred)

print(np.round(np.mean(L),3))
print(np.round(np.std(L),3))
