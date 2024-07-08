import numpy as np
from scipy.stats import ttest_rel

# dataset x model x metric x fold

results_data = np.load("scores.npy")

# Model x model x metric x dataset

t_stat = np.zeros((3,3,4,3))
p_value = np.zeros((3,3,4,3))

bool_result = np.zeros((3,3,4,3), dtype=bool)
is_alpha_ok = np.zeros((3,3,4,3), dtype=bool)
bool_result_with_alpha = np.zeros((3,3,4,3), dtype=bool)

for i in range(3):
    for j in range(3):
        for p in range(4):
            for k in range(3):
                t_stat[i,j,p,k] = ttest_rel(results_data[k,i,p,:], results_data[k,j,p,:])[0]
                p_value[i,j,p,k] = ttest_rel(results_data[k,i,p,:], results_data[k,j,p,:])[1]
                if t_stat[i,j,p,k]>0:
                    bool_result[i,j,p,k] = True
                if p_value[i,j,p,k]<0.1:
                    is_alpha_ok[i,j,p,k] = True

bool_result_with_alpha = bool_result * is_alpha_ok


for i in range(3):
    for j in range(4):
        print(f"For the dataset {i+1} and the metric {j+1}")
        print(bool_result[:,:,j,i])
