""" Inertia example """
import numpy as np
from scipy.linalg import ldl
import casadi as cas

x = cas.SX.sym('x')
y = cas.SX.sym('y')
u = cas.SX.sym('u')
J = 1/3 * x**3 - x + y*y
L = J + u * (x-y)
var = cas.vertcat(x, y, u)
kkt_res = cas.jacobian(L, var).T
hessian = cas.jacobian(kkt_res, var)

kkt_fun = cas.Function('kkt', [var], [kkt_res])
hessian_fun = cas.Function('hessian', [var], [hessian])

newton = cas.rootfinder('newton', 'newton', kkt_fun)

retval = newton(cas.DM([0.3, 0.2, 0.1]))
hessian_val = hessian_fun(retval)
print(f'{retval=}')
print(f'{hessian_val=}')
lu, d, perm = ldl(np.array(hessian_val))
print(f'{np.diag(d)=}')

retval = newton(cas.DM([-3, -2, 0.1]))
hessian_val = hessian_fun(retval)
print(f'{retval=}')
print(f'{hessian_val=}')
lu, d, perm = ldl(np.array(hessian_val))
print(f'{np.diag(d)=}')
