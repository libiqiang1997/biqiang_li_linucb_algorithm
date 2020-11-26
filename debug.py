import numpy as np


x = 0.13190491559486367 * np.array([0.44999316, 0.80535737])
y = x.transpose()
z = y.dot(x)
print(x)
print(y)
print(z)
