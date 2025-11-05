import numpy as np


arr = np.array([
    [0, 0.2, 0],
    [1, 2, 3],
    [0, 0.01, 0]])

mask = (arr > 0.05).any(axis=1)

print(mask)