# benchmark eigen(c++) and python for simple matrix manipulation
from time import time
import numpy as np

I = 100; J = 100; K = 100
fill = np.linspace(0,99,100).reshape(10,10)

t0 = time()
for iter in range(100):
    mtx = np.zeros((I,J))
    for i in range(int(I/10)):
        for j in range(int(J/10)):
            for k in range(K):
                mtx[10*i:10*(i+1),10*j:10*(j+1)] += k*fill @ fill if (i+j)%2==0 else 1
print(f'dt = {time()-t0}')
print(np.linalg.norm(mtx))

