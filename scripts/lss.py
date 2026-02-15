# Illustration for Local Symplectic Surgery
import numpy as np
import casadi as cas


class LSS:
    """ Use local symplectic surgery to solve a demo problem"""

    def __init__(self):
        x = cas.Sx.sym('x', 1, 1)
        y = cas.Sx.sym('y', 1, 1)
        J = self.J(x, y)
        J_fun = cas.Function('J', [x, y], [J])
        Jx = cas.jacobian(J, x)
        Jy = cas.jacobian(J, y)
        # min_x J, min_y -J
        w = cas.vertcat(Jx, -Jy)
        z = cas.vertcat(x, y)
        Dw = cas.jacobian(w, z)
        I = np.eye(2)
        C = 0.1
        dzdt_lss = -0.5 * (w + Dw.T @ cas.inv(Dw.T @ Dw + C*I) @ Dw.T @ w)
        dzdt_gd = Dw

    def J(self, x, y):
        """ Return the objective function """
        return x * y + (x-0.5)**2 + (y-0.5)**2

    def get_trajectory(self, x0, y0, method='gd'):
        if method == 'gd':
            # gradient descent, follow dzdt = -Dw
            pass
        elif method == 'lss':
            # LSS, follow dzdt
            pass
        return

    def plot(self):
        # plot 3 things
        # 1. The cost landscape, from [-1,1], use level lines with a gentle color grading
        # 2. The local saddle point (Nash in zero sum) and local minimals
        #    (min_{x,y} J, not a solution) as dots and crosses
        # 3. Trajectory of x,y when they folow gd and lss. use a tuneable dt


if __name__ == '__main__':
    main = LSS()
