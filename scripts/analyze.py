import pickle as p
import numpy as np
import matplotlib.pyplot as plt
from BimodalConvergence import BimodalConvergence
from scipy.stats import gaussian_kde

# check the residual of various points

#data = {'belief_support':main.belief_support, 'belief_weight':main.belief_weight, 'belief_support_residual':main.belief_support_residual, 'belief_support_cost':main.belief_support_cost}
with open('manifold_particles.p', 'rb') as f:
    data = p.load(f)

main = BimodalConvergence()

# ---- plot final particles ----
x_ref_vec = []
print(f"particles: {len(data['belief_support_cost'])}")
for u_ref in data['belief_support']:
    x_ref = main.rollout(main.x0, u_ref)
    x_ref_vec.append(x_ref)
x_ref_vec = np.array(x_ref_vec)

#mask = np.linalg.norm(x_ref_vec[:,-1,:,0], axis=1) < 0.4
#print(data['belief_support_residual'])
#print(data['belief_support_cost'])

points = x_ref_vec[:, -1, :, 0]
# TODO screen for saddle points with second order conditions
#mask = np.linalg.norm(points, axis = 1) > 0.3
#points = points[mask]
plt.plot(points[:, 0], points[:, 1], '*')
plt.show()


def kernel(x, y):
    return np.exp(-10 * np.linalg.norm(x - y)**2 / np.linalg.norm(y))


# ---- Plot the heatmap --- of final particles
def plotHeatmap(points):
    x_edges = np.linspace(-1.5, 1.5, 100)
    y_edges = np.linspace(-1.5, 1.5, 100)
    X, Y = np.meshgrid(x_edges, y_edges)
    #Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    Z = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            for p in points:
                point = np.array([X[i, j], Y[i, j]])
                Z[i, j] += kernel(point, p)

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(Z.T,
               origin='lower',
               extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
               cmap='hot',
               aspect='auto')
    plt.colorbar(label='Density')
    plt.xlabel('Agent 0, x[0]')
    plt.ylabel('Agent 1, x[0]')
    plt.title('')
    #plt.show()


plt.ioff()
for k in range(len(data['particle_history'])):
    particles = data['particle_history'][k]
    x_ref_vec = []
    mask = main.get_mask_for_pd_hessian(particles)
    particles = particles[mask]
    for theta in particles:
        dim_u = (main.T, main.N, main.m)
        u_ref = theta.reshape(dim_u)
        x_ref = main.rollout(main.x0, u_ref)
        x_ref_vec.append(x_ref)
    points = np.array(x_ref_vec)[:, -1, :, 0]
    print(f'step {k}')
    plotHeatmap(points)
    plt.savefig(f'./pics/manifold_hessian_{k}')
plt.show()
