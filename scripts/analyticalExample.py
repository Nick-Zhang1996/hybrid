# solve the joint equation from the analytical example
import numpy as np
from scipy import optimize

p1 = 1; p2 = -1
R = 1

def res(xi,xj,ui,lambdai):
    dLi_dxi = - 2*(xi-p1) * np.exp(-(xi-p1)**2 - (xj-p2)**2 ) \
              - 2*(xi-p2) * np.exp(-(xi-p2)**2 - (xj-p1)**2 )
    dLi_dui = 2*R*ui - lambdai
    return (dLi_dxi, dLi_dui)

def fun(val):
    u1,u2,lambda1,lambda2 = val
    x1 = u1; x2 = u2 # assume x0 = 0 for both agents
    res1 = res(x1,x2,u1,lambda1)
    res2 = res(x2,x1,u2,lambda2)
    return res1+res2

# 0,0,0,0   -> sol1
# 1,-1,2,-2 -> sol2
# -1,1,-2,2 -> sol3
guess = [1,-1,0,0]
guess = [0.1,-0.1,0,0]
guess = [-1,1,0,0]
print(fun(guess))
sol = optimize.root(fun, guess)
print(sol)
print(sol.x)
