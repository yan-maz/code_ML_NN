import numpy as np


err = np.load('knn_error.npy')
err2 = np.load('Rdm_error.npy')


for i in range(3):
    print(f'Knn mean {np.round(np.mean(err[i]),3)} and std {np.round(np.std(err[i]),3)} for dataset number {i}')
    print(f'Rdm mean {np.round(np.mean(err2[i]),3)} and std {np.round(np.std(err2[i]),3)} for dataset number {i}')