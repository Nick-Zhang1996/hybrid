import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from BimodalConvergence import BimodalConvergence

def Jfi( x_T, i):
    return    np.log(2*( (x_T[0] - main.p1).T @ main.J_Q @ (x_T[0] - main.p1)  \
            +  (x_T[1] - main.p2).T @ main.J_Q @ (x_T[1] - main.p2) )+1) \
            + np.log(2*( (x_T[0] - main.p2).T @ main.J_Q @ (x_T[0] - main.p2)  \
            +  (x_T[1] - main.p1).T @ main.J_Q @ (x_T[1] - main.p1) )+1)

main = BimodalConvergence()
x = np.linspace(-4,4,100)
y = np.linspace(-4,4,100)
[xx,yy] = np.meshgrid(x,y)
zz = np.zeros_like(xx)
for i in range(zz.shape[0]):
    for j in range(zz.shape[1]):
        x0 = np.vstack( [ xx[i,j], 0 ] )
        x1 = np.vstack( [ yy[i,j], 0 ] )
        x_T = np.hstack([x0,x1]).T
        zz[i,j] = Jfi(x_T, 0)
idx = np.argmin(zz.flatten())
x_val = xx.flatten()[idx]
y_val = yy.flatten()[idx]
print(x_val, y_val, zz.flatten()[idx])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx,yy,zz)
plt.show()
