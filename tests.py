import numpy as np
import matplotlib.pyplot as plt
from functions import covariance

rng = np.random.default_rng(1)
x = rng.normal(size=(8,1)) * 10
y = 2*x + 1 + rng.normal(size=(8,1)) * 5

A = np.hstack([x,y])

cov_matrix = covariance(A)
evalues, evectors = np.linalg.eig(cov_matrix)
evectors = evectors/np.sum(evectors**2,axis=0)**0.5 #normalize
eval_order = np.argsort(evalues)[::-1]
evalues = evalues[eval_order]
evectors = evectors[:,eval_order]
new_x_axis = evectors[:,0]
new_y_axis = evectors[:,1]
new_x_coordinates = A @ new_x_axis
new_y_coordinates = A @ new_y_axis
x_projections = np.outer(new_x_coordinates,new_x_axis)

fig, ax =plt.subplots()
ax.scatter(x,y,label='A')
ax.scatter(new_x_coordinates, new_y_coordinates, label='AV')
ax.scatter(*x_projections.T,label='New X Coordinates',marker='x')
ax.legend()
ax.set_aspect('equal')
plt.show()
print()