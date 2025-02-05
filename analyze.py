import pickle as p
import numpy as np
import matplotlib.pyplot as plt
from BimodalConvergence import BimodalConvergence

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

mask = np.linalg.norm(x_ref_vec[:,-1,:,0], axis=1) < 0.4
print(mask)
print(data['belief_support_residual'])
print(data['belief_support_cost'])
breakpoint()

