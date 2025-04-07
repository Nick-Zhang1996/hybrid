# test off-the-shelf solvers
# Our problem:
# root finding problem
# format1 : solve r(x,u,lambda, mu) = 0, have access to gradient

# reformulate as SQP problem
# format2 : minimize |r|^2 -> SQP
# r = Ax + b -> r^2 = xT AT A x + 2 bT Ax

# mosek:
#       no NLP, can't do SQP, uncompatible
#       Technically we can do our own SQP
# gurobi:
#       v9 supports nonconvexity, but only quadratic models
# cvxopt -> only quadratic programming
# FORCES PRO - NLP by SQP
from utilities.util import *
import ipopt


class Example(object):
    def __init__(self):
        pass

    def res_i(xi,xj,ui,lambdai):
        dLi_dxi = - 2*(xi-p1) * np.exp(-(xi-p1)**2 - (xj-p2)**2 ) \
                  - 2*(xi-p2) * np.exp(-(xi-p2)**2 - (xj-p1)**2 )
        dLi_dui = 2*R*ui - lambdai
        return (dLi_dxi, dLi_dui)

    def objective(self, x):
        x1, x2, u1, u2,lambda1,lambda2 = x
        res1 = res(x1,x2,u1,lambda1)
        res2 = res(x2,x1,u2,lambda2)
        obj = np.sum(np.array(res1+res2)**2)
        return obj
    def gradient(self, x):
        jac = jacobianNumerical(self.objective,x,dim=r0.shape[0])
        return jac

    def constraints(self, x):
        x1, x2, u1, u2,lambda1,lambda2 = x
        return np.array( (x1-u1, x2-u2) )

    def jacobian(self, x):
        # for the constraints
        return np.array([[1,0,-1,0],[0,1,0,-1]])

    def hessian(self, x, lagrange, obj_factor):
        hes = obj_factor * hessianNumerical(self.objective, x)
        # convert to lower triangular matrix, flattened
        hs = sps.coo_matrix(np.tril(np.ones((hes.shape[0], hes.shape[0]))))
        return hes[hs.row, hs.col]
    def intermediate(
         self,
         alg_mod,
         iter_count,
         obj_value,
         inf_pr,
         inf_du,
         mu,
         d_norm,
         regularization_size,
         alpha_du,
         alpha_pr,
         ls_trials
         ):
         print "Objective value at iteration #%d is - %g" % (iter_count, obj_value)


x0 = [0,0,0,0]
lb = [-10,-10,-10,-10]
ub = [10,10,10,10]
cl = [0,0]
cu = [0,0]
nlp = ipopt.problem(n=6,m=2, problem_obj = Example(), lb = -10, ub = ub, cl = cl, cu = cu)
nlp.addOption('mu_strategy', 'adaptive')
nlp.addOption('tol', 1e-7)
x, info = nlp.solve(x0)
print(x)
print(info)
