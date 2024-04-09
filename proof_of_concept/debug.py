# debug PerlinNoise gradient issue
import matplotlib.pyplot as plt
from Problem import *
from Solver import *


problem = PerlinNoise()

fig = plt.figure()
ax = Axes3D(fig)
N = 50
val = [0.6,0.6]
xx_vec, yy_vec = np.meshgrid(np.linspace(-0.1,0.1,N),np.linspace(-0.01,0.01,N))
xx_vec += val[0]
yy_vec += val[1]
zz_vec = []
jac_vec = []
J = problem.jacobian(val)
for x,y in zip(xx_vec.flatten(), yy_vec.flatten()):
    zz_vec.append(problem.evaluate(np.array([x,y])))
    jac_vec.append(problem.evaluate(val)+ J@(np.array([x,y])-val).T)

ax.plot_surface(np.array(xx_vec),np.array(yy_vec),np.array(zz_vec).reshape(N,N),color=(1.0,0,0))
ax.plot_surface(np.array(xx_vec),np.array(yy_vec),np.array(jac_vec).reshape(N,N),color=(0,1.0,0))

plt.show()
