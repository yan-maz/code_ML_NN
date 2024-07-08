import numpy as np

array1 = np.ones((3,5,7))

a = np.sum(array1,axis = 0)

b = np.sum(array1,axis = 1)

c = np.sum(array1,axis = 2)

print(a)
print(b)
print(c)