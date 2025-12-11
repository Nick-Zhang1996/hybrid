""" Construct a residual function from constrained optimization"""
from casadi import *

# J(x) = x[0] + x[1]
# g(x) = || x ||_2 = 1

x = SX.sym('x', 2)
lamda = SX.sym('lamda', 1)
J = x[0] + x[1]
g = norm_2(x)

L = J + lamda.T @ g
stationarity = jacobian(L, x)
constraints = g - 1

decision_vars = vertcat(x, lamda)
kkt_residual = vertcat(stationarity.T, constraints)
r = Function('r', [decision_vars], [kkt_residual])

x_val = DM([0.1, 0.2])
lamda_val = DM([0.1])
decision_var_val = vertcat(x_val, lamda_val)
r_val = r(decision_var_val)
print(r_val)

# solving with existing solver
solver = rootfinder('solver', 'newton', r)
res = solver([0.1, 0.1, 0.1])
print(res)

# solving manually
cg = CodeGenerator('gen.c', {'with_header': True})
cg.add(r)
cg.generate()
