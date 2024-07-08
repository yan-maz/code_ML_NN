import numpy as np
from scipy.stats import ttest_rel
from scipy.stats import rankdata
from scipy.stats import ranksums

results = np.load("results.npy")
results_mean = np.mean(results, axis = 2)

data = rankdata(results_mean, axis = 1)




t_stat = np.zeros((3,3))
p_value = np.zeros((3,3))
bool_result = np.zeros((3, 3), dtype=bool)
is_alpha_ok = np.zeros((3, 3), dtype=bool)
bool_result_with_alpha = np.zeros((3, 3), dtype=bool)

for i in range(3):
    for j in range(3):
        t_stat[i,j] = ranksums(data[:,i], data[:,j])[0]
        p_value[i,j] = ranksums(data[:,i], data[:,j])[1]
        if t_stat[i,j]>0:
            bool_result[i,j] = True
        if p_value[i,j]<0.05:
            is_alpha_ok[i,j] = True


bool_result_with_alpha = bool_result * is_alpha_ok



print(t_stat)
print(p_value)
print(bool_result)
print(is_alpha_ok)
print(bool_result_with_alpha)

print(np.mean(data, axis = 0))

