import matplotlib.pyplot as plt
import pickle
with open('convergence.p','rb') as f:
    residual_vec_vec = pickle.load(f)

diverge_count = 0
for residual_vec in residual_vec_vec:
    if (len(residual_vec) == 50 and residual_vec[30]<1e-2):
        plt.plot(residual_vec,'-*')
    else:
        diverge_count += 1

with open('convergence.p', 'wb') as f:
    pickle.dump(residual_vec_vec,f)

print(f'diverge count {diverge_count}')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Residual (exp)')
plt.show()


