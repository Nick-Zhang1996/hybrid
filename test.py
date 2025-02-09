import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from BimodalConvergence import BimodalConvergence

def Jfi( x_T, i):
    return    np.log(2*( (x_T[0] - main.p1).T @ main.J_Q @ (x_T[0] - main.p1)  \
            +  (x_T[1] - main.p2).T @ main.J_Q @ (x_T[1] - main.p2) )+3) \
            + np.log(2*( (x_T[0] - main.p2).T @ main.J_Q @ (x_T[0] - main.p2)  \
            +  (x_T[1] - main.p1).T @ main.J_Q @ (x_T[1] - main.p1) )+3)

main = BimodalConvergence()
x = np.linspace(-3,3,100)
y = np.linspace(-3,3,100)
[xx,yy] = np.meshgrid(x,y)
zz = np.zeros_like(xx)
for i in range(zz.shape[0]):
    for j in range(zz.shape[1]):
        '''
        x0 = np.vstack( [ xx[i,j], 0 ] )
        x1 = np.vstack( [ yy[i,j], 0 ] )
        x_T = np.hstack([x0,x1]).T
        zz[i,j] = Jfi(x_T, 0)
        zz[i,j] = -np.exp(-(xx[i,j]-1)**2 - (yy[i,j]+1)**2) -np.exp(-(xx[i,j]+1)**2 - (yy[i,j]-1)**2)
        '''
        f = xx[i,j]**2 + yy[i,j]**2 -1
        zz[i,j] = -np.exp(-f**2)

idx = np.argmin(zz.flatten())
x_val = xx.flatten()[idx]
y_val = yy.flatten()[idx]
print(x_val, y_val, zz.flatten()[idx])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx,yy,zz)
plt.show()
exit(0)

xx = np.linspace(-3,3)
yy = -xx + 3

'''
zz = np.zeros_like(xx)
for i in range(xx.shape[0]):
    x0 = np.vstack( [ xx[i], 0 ] )
    x1 = np.vstack( [ yy[i], 0 ] )
    x_T = np.hstack([x0,x1]).T
    zz[i] = Jfi(x_T, 0)
'''
zz = -np.exp(-(xx-1)**2 - (yy+1)**2) -np.exp(-(xx+1)**2 - (yy-1)**2)
plt.plot(zz)
plt.show()
