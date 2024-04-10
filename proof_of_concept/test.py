from scipy.optimize import fsolve
from math import sin,cos
import numpy as np
from Problem import *
A = 2
B = 10
C = 0.4
def fun(x):
    x1,x2,u = x
    return ( (x1-0.15)**2+(x2-0.05)**2-0.09, 2*A*x1 + C*cos(B*x1)*B + 2*(x1-0.15)*u, -C*B*sin(B*x2)+2*(x2-0.05)*u)

x1,x2,u = fsolve(fun,(-0.11,0.26,1.0))
print(x1,x2,u)
hx = lambda x:(x[0]-0.15)**2+(x[1]-0.05)**2-0.3**2
print(hx([x1,x2]))

problem = ParabolaWithSineNoise2D()
fun_x = problem.evaluate(np.array([x1,x2]))
print(fun_x)


breakpoint()
