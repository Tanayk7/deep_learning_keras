import numpy as np

arr = np.array([1,2,2])
arr2 = np.array([1,2,2])
l = [arr,arr2]
arr3 = np.asarray(l)
print(arr3)
print(arr3.shape)