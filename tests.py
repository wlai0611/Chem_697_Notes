import numpy as np
import matplotlib.pyplot as plt
from functions import covariance

rng = np.random.default_rng(1)
x = rng.normal(size=(8,1)) * 10
y = 2*x + 1 + rng.normal(size=(8,1)) * 5

A = np.hstack([x,y])

plt.scatter(x,y)
plt.show()

cov_matrix = covariance(A)
evalues, evectors = np.linalg.eig(cov_matrix)
evectors = evectors/np.sum(evectors**2,axis=0)**0.5 #normalize
new_coordinates = A @ evectors

fig, ax =plt.subplots()
ax.scatter(x,y,label='A')
ax.scatter(new_coordinates[:,0], new_coordinates[:,1], label='AV')
ax.legend()
ax.set_aspect('equal')
plt.show()
print()