from strlearn.streams import StreamGenerator
from strlearn.evaluators import TestThenTrain
from sklearn.metrics import balanced_accuracy_score
from strlearn.ensembles import SEA
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

clf = [GaussianNB(),MLPClassifier()]

stream = StreamGenerator(n_chunks=300, n_drifts=3, chunk_size = 200, weights=[0.95, 0.05], concept_sigmoid_spacing=999) #, incremental = False, n_features = 8, n_informative = 8, n_redundant = 0)

evaluator = TestThenTrain(metrics=(balanced_accuracy_score))

evaluator.process(stream, clf)



plt.plot(evaluator.scores[0], c = 'r')

plt.plot(evaluator.scores[1], c = 'b')

plt.savefig('fig')

#for chunk_id in range(n_chunks):
# #    X_chunk, y_chunk = X[chunk_id * chunk_size : (chunk_id+1) * chunk_size], y[chunk_id * chunk_size : (chunk_id+1) * chunk_size]

# print(evaluator)



# from strlearn.evaluators import TestThenTrain
# from strlearn.ensembles import SEA
# from strlearn.utils.metrics import bac, f_score
# from strlearn.streams import StreamGenerator
# from sklearn.naive_bayes import GaussianNB


# clf = SEA(base_estimator=GaussianNB(), MLPClassifier())
# evaluator = TestThenTrain(metrics=(bac))

# evaluator.process(stream, clf)
# print(evaluator.scores)