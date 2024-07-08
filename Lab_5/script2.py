import numpy as np
from scipy.stats import ttest_rel

results_data_1 = np.load("results.npy")[0]

t_stat = np.zeros((3,3))
p_value = np.zeros((3,3))

bool_result = np.zeros((3, 3), dtype=bool)
is_alpha_ok = np.zeros((3, 3), dtype=bool)
bool_result_with_alpha = np.zeros((3, 3), dtype=bool)
for i in range(3):
    for j in range(3):
        t_stat[i,j] = ttest_rel(results_data_1[i], results_data_1[j])[0]
        p_value[i,j] = ttest_rel(results_data_1[i], results_data_1[j])[1]
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

print(np.mean(results_data_1[0]))
print(np.mean(results_data_1[1]))
print(np.mean(results_data_1[2]))