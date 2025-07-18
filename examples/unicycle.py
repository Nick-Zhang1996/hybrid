""" Unicycle Dynamics to replicate experiment in https://arxiv.org/pdf/2002.04354 """

from math import sin, cos
import logging

import sympy
import numpy as np
import matplotlib.pyplot as plt

from residual_game import ResidualGame, ResidualGameConfig, ResidualGameConfig
from utilities import symbolic_dynamics

logger = logging.getLogger("Unicycle")


def wrap(x: float):
    """ Wrap angle (rad) to [-pi,pi] """
    return (x + np.pi) % (2 * np.pi) - np.pi


class Unicycle(ResidualGame):
    """ 3/5 agent trajectory planning with unicycle dynamics"""

    def __init__(self, config: ResidualGameConfig, agent_count: int):
        super().__init__(config)
        self.N = agent_count
        self.T = 20  # 100
        self.dt = 0.1

        # dimensions of x and u for a single agent
        self.n = 4
        self.m = 2

        self.dim_theta = self.T * self.N * self.m
        self.covariance_mtx = np.eye(self.dim_theta) * 1.0

        # bounds for visualization
        self.visual_x_lim = [-2.5, 2.5]
        self.visual_y_lim = [-2, 30]

        # The game is a planning task for N agents
        # where each agent starts unidistance on a circle
        # and the target pose is over the diameter

        # Initial position
        # x = (x, y, heading, v)
        # u = (a, omega)
        radius = 2.5
        phase_vec = np.linspace(0, 2 * np.pi, self.N)
        xx = radius * np.cos(phase_vec)
        yy = radius * np.sin(phase_vec)
        heading_vec = -wrap(phase_vec + np.pi)
        vv = np.ones_like(phase_vec)

        xx_ref = radius * np.cos(phase_vec + np.pi)
        yy_ref = radius * np.sin(phase_vec + np.pi)

        self.J_x_ref_fun = lambda i: np.array(
            [xx_ref[i], yy_ref[i], heading_vec[i], vv[i]])
        # Reference cost
        self.J_Qr = np.eye(self.n) * 0.5
        # State cost
        self.J_Q = np.eye(self.n) * 0.1
        # Control cost
        self.J_R = np.eye(self.m) * 0.1

        # Initial states, [N, n]
        self.x0 = np.hstack([xx, yy, heading_vec, vv])
        self.guess = np.zeros((self.T, self.N, self.m))

    def setup(self):
        if self.config.USE_CPP or self.config.CPP_DEBUG:
            raise NotImplementedError

    def _visualize(self, u: np.ndarray, x: np.ndarray | None = None):
        """ Visualize state trajectory given experiment type
        Args:
            u: control trajectory, [T, N, m]
            x: state trajectory starting from to,
              [Any, N, n], if None, will be rollout from u
        Return:
            plotted fig object
        """
        if (u is None):
            x = np.vstack(
                [self.x0[np.newaxis, :, :],
                 self.rollout(self.x0, u)])
        fig, ax = plt.subplots()
        for i in range(self.N):
            # (x,y,heading,v)
            xx = x[:, i, 0]
            yy = x[:, i, 1]
            plt.plot(xx, yy, '*-')

        ax.set_aspect('equal', adjustable='box')
        return fig

    def _animation(self,
                   u: np.ndarray,
                   x: np.ndarray | None = None,
                   gif_prefix: str = ''):
        raise NotImplementedError

    def J(self, x_k: np.ndarray, u_k_i: np.ndarray, i: int) -> float:
        """ Step cost function for agent i.
        Args:
            x_k: [N, n] *all* agent state at this step (k), (x, y, heading, v)
            u_k_i: [m] control for agent i at this step (k), (a, omega)
                a=dv_dt is acceleration
                omega=dheading_dt is angular acceleration
            i: agent id, starts from 0
        Return:
            cost for agent i at this step (k)
            """
        # quadratic state error penalty
        dx = (x_k[i] - self.J_x_ref_fun(i))
        cost = dx.T @ self.J_Qr @ dx + x_k[i].T @ self.J_Q @ x_k[
            i] + u_k_i.T @ self.J_R @ u_k_i
        # collision cost, soft constraint as in reference paper
        d = 0.1

        def collision_cost(i, j):
            dx = x_k[i, 0] - x_k[j, 0]
            dy = x_k[i, 1] - x_k[j, 1]
            return max(0, d**2 - dx**2 - dy**2)

        cost += sum(collision_cost(i, j) for j in range(self.N) if j != i)
        return cost

    def dJi_dxi(self, x_k, u_k_i, i):
        """ Step cost gradient w.r.t. x_i
        Args:
            x_k: [N, n] *all* agent state at this step (k), (x, y, heading, v)
            u_k_i: [m] control for agent i at this step (k), (a, omega)
                a=dv_dt is acceleration
                omega=dheading_dt is angular acceleration
            i: agent id, starts from 0
            j: agent id, starts from 0
        Return:
            [1, n] Partial derivative
        """
        val = 2 * (x_k[i] -
                   self.J_x_ref_fun(i)).T @ self.J_Qr + 2 * x_k[i].T @ self.J_Q
        d = 0.1

        def collision_cost_grad(i, j):
            dx = x_k[i, 0] - x_k[j, 0]
            dy = x_k[i, 1] - x_k[j, 1]
            cost = d**2 - dx**2 - dy**2
            return np.array([[2 * dx, 2 * dy, 0, 0]]) if cost > 0 else 0

        for j in range(self.N):
            if i == j:
                continue
            val += collision_cost_grad(i, j)
        return val

    def dJi_dxj(self, x_k, u_k_i, i, j):
        """ Step cost gradient w.r.t. x_j
        Args:
            x_k: [N, n] *all* agent state at this step (k), (x, y, heading, v)
            u_k_i: [m] control for agent i at this step (k), (a, omega)
                a=dv_dt is acceleration
                omega=dheading_dt is angular acceleration
            i: agent id, starts from 0
            j: agent id, starts from 0
        Return:
            [1, n] Partial derivative
        """
        val = np.zeros((1, self.n))
        d = 0.1

        def collision_cost_grad(i, j):
            dx = x_k[i, 0] - x_k[j, 0]
            dy = x_k[i, 1] - x_k[j, 1]
            cost = d**2 - dx**2 - dy**2
            return np.array([[-2 * dx, -2 * dy, 0, 0]]) if cost > 0 else 0

        for j in range(self.N):
            if i == j:
                continue
            val += collision_cost_grad(i, j)
        return val

    def dJi_du(self, x_k, u_k_i, i):
        return 2 * u_k_i.T @ self.J_R

    def dJi_dxi_dxi(self, x_k, u_k_i, i):
        """ Step cost second order derivative w.r.t. x_i
        Args:
            x_k: [N, n] *all* agent state at this step (k), (x, y, heading, v)
            u_k_i: [m] control for agent i at this step (k), (a, omega)
                a=dv_dt is acceleration
                omega=dheading_dt is angular acceleration
            i: agent id, starts from 0
            j: agent id, starts from 0
        Return:
            [1, n] Partial derivative
        """
        val = 2 * self.J_Qr + 2 * self.J_Q
        d = 0.1

        def collision_cost_hess(i, j):
            dx = x_k[i, 0] - x_k[j, 0]
            dy = x_k[i, 1] - x_k[j, 1]
            cost = d**2 - dx**2 - dy**2
            return np.diag([[2, 2, 0, 0]]) if cost > 0 else 0

        for j in range(self.N):
            if i == j:
                continue
            val += collision_cost_hess(i, j)
        return val

    def dJi_dxi_dxj(self, x_k, u_k_i, i, j):
        """ Step cost second ordder derivative w.r.t. x_i, then x_j
        Args:
            x_k: [N, n] *all* agent state at this step (k), (x, y, heading, v)
            u_k_i: [m] control for agent i at this step (k), (a, omega)
                a=dv_dt is acceleration
                omega=dheading_dt is angular acceleration
            i: agent id, starts from 0
            j: agent id, starts from 0
        Return:
            [1, n] Partial derivative
        """
        return np.zeros((self.n, self.n))

    def dJi_dxj_dxj(self, x_k, u_k_i, i, j):
        """ Step cost second order derivative w.r.t. x_j
        Args:
            x_k: [N, n] *all* agent state at this step (k), (x, y, heading, v)
            u_k_i: [m] control for agent i at this step (k), (a, omega)
                a=dv_dt is acceleration
                omega=dheading_dt is angular acceleration
            i: agent id, starts from 0
            j: agent id, starts from 0
        Return:
            [1, n] Partial derivative
        """
        val = 2 * self.J_Qr + 2 * self.J_Q
        d = 0.1

        def collision_cost_hess(i, j):
            dx = x_k[i, 0] - x_k[j, 0]
            dy = x_k[i, 1] - x_k[j, 1]
            cost = d**2 - dx**2 - dy**2
            return np.diag([[2, 2, 0, 0]]) if cost > 0 else 0

        for j in range(self.N):
            if i == j:
                continue
            val += collision_cost_hess(i, j)
        return val

    def dJi_dudu(self, x_k, u_k_i, i):
        return 2 * self.J_R

    def f(self, x: np.ndarray, u: np.ndarray, i: int) -> np.ndarray:
        """ Dynamics funciton, gives x(state) at next time step 
        Args:
            x: [n] states of agent i
                x = (x, y, heading, v)
            u: [m] control of agent i
                u = (a, omega)
            i: agent index 
        Return:
            States (x) [n] at next time step

        """
        # dxdt = (v*cos(heading),v*sin(heading), omega, a )
        dxdt = np.array([x[3] * cos(x[2]), x[3] * sin(x[2]), u[1], u[0]])
        return x + dxdt * self.dt

    def df_dx(self, x: np.ndarray, u: np.ndarray, i: int) -> np.ndarray:
        """ Dynamics derivative
        Args:
            x: [n] states of agent i
                x = (x, y, heading, v)
            u: [m] control of agent i
                u = (a, omega)
            i: agent index 
        Return:
            States (x) [n] at next time step
        """
        x0, x1, x2, x3 = x
        u0, u1 = u
        dfdx = np.array([[0, 0, -x3 * sin(x2), cos(x2)],
                         [0, 0, x3 * cos(x2), sin(x2)], [0, 0, 0, 0],
                         [0, 0, 0, 0]])
        return np.eye(4) + dfdx * self.dt

    def df_du(self, x: np.ndarray, u: np.ndarray, i: int) -> np.ndarray:
        """
        Args:
            x: [n] states of agent i
                x = (x, y, heading, v)
            u: [m] control of agent i
                u = (a, omega)
            i: agent index 
        Return:
            States (x) [n] at next time step
        """
        dfdu = np.array([[0, 0], [0, 0], [0, 1], [1, 0]])
        return dfdu * self.dt

    def get_symbolic_dynamics(self):
        """Helper functions for dynamics"""
        dyn = symbolic_dynamics.SymbolicDynamics(n=4, m=2)
        # x = (x, y, heading, v)
        x = dyn.x[0]
        y = dyn.x[1]
        heading = dyn.x[2]
        v = dyn.x[3]
        # u = (a, omega)
        a = dyn.u[0]
        omega = dyn.u[1]
        dyn.f = [v * sympy.cos(heading), v * sympy.sin(heading), omega, a]
        dyn.symDerF()
        print(f'{dyn.dfdx=}')
        print(f'{dyn.dfdu=}')


if __name__ == "__main__":
    _config = ResidualGameConfig()
    print(_config)
    main = Unicycle(_config, agent_count=3)
    #main.get_symbolic_dynamics()
    main.setup()
    u_ref, full_x_ref, has_converged = main.solve()
    main.final()
