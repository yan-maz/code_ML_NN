from sklearn.datasets import make_moons
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_circles
import numpy as np
from script1 import RandomClassifier
from script2 import kNN
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

X, y = make_circles()
dataset = [make_circles(),make_gaussian_quantiles() ,make_moons()]
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=36851234)

error1 = np.zeros((3,10))
error2 = np.zeros((3,10))

for j in range(3):
    X, y = dataset[j]
    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        mod1 = RandomClassifier()
        mod2 = kNN()
        mod1.fit(X[train_index], y[train_index])
        mod2.fit(X[train_index], y[train_index])
        error1[j,i] = accuracy_score(y[test_index], mod1.predict(X[test_index]))
        error2[j,i] = accuracy_score(y[test_index], mod2.predict(X[test_index]))

print(error1)
np.save('Rdm_error', error1)
print(error2)
np.save('knn_error', error2)