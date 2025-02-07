import pickle as p
import numpy as np
import matplotlib.pyplot as plt
from BimodalConvergence import BimodalConvergence
from scipy.stats import gaussian_kde

# check the residual of various points

#data = {'belief_support':main.belief_support, 'belief_weight':main.belief_weight, 'belief_support_residual':main.belief_support_residual, 'belief_support_cost':main.belief_support_cost}
with open('particles.p', 'rb') as f:
    data = p.load(f)

main = BimodalConvergence()

x_ref_vec = []
for u_ref in data['belief_support']:
    x_ref = main.rollout(main.x0,u_ref)
    x_ref_vec.append(x_ref)
x_ref_vec = np.array(x_ref_vec)

#mask = np.linalg.norm(x_ref_vec[:,-1,:,0], axis=1) < 0.4
#print(data['belief_support_residual'])
#print(data['belief_support_cost'])

points = x_ref_vec[:,-1,0]
# TODO screen for saddle points with second order conditions
mask = np.linalg.norm(points, axis = 1) > 0.3
points = points[mask]
plt.plot(points[:,0], points[:,1], '*')
plt.show()

# Perform kernel density estimation (KDE)
kde = gaussian_kde(points.T)
#x_edges = np.linspace(points[:, 0].min(), points[:, 0].max(), 30)
#y_edges = np.linspace(points[:, 1].min(), points[:, 1].max(), 30)
x_edges = np.linspace(-1.5, 1.5, 100)
y_edges = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(x_edges, y_edges)
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

# Plot the heatmap
plt.figure(figsize=(8, 6))
plt.imshow(Z.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap='hot', aspect='auto')
plt.colorbar(label='Density')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Smoothed Heatmap of Point Distribution using KDE')
plt.show()

