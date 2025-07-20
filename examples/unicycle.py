""" Unicycle Dynamics to replicate experiment in https://arxiv.org/pdf/2002.04354 """

from math import sin, cos
import logging
import itertools

import sympy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

from residual_game import ResidualGame, ResidualGameConfig, ResidualGameConfig
from src.build.unicycle import Unicycle as cpp_Unicycle
from utilities import symbolic_dynamics
from utilities.util import jacobianNumerical

logger = logging.getLogger("Unicycle")
logger.setLevel(logging.DEBUG)

logging.basicConfig(level=logging.WARNING)

_DEBUG = False
_CPP_DEBUG = True
""" File-level debug flag"""


def wrap(x: float):
    """ Wrap angle (rad) to [-pi,pi] """
    return (x + np.pi) % (2 * np.pi) - np.pi


class Unicycle(ResidualGame):
    """ 3/5 agent trajectory planning with unicycle dynamics"""

    def __init__(self, config: ResidualGameConfig, agent_count: int):
        super().__init__(config)
        self.N = agent_count
        self.T = 100
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

        self.collision_diameter = 0.2
        # Initial position
        # x = (x, y, heading, v)
        # u = (a, omega)
        radius = 2.5
        phase_vec = np.linspace(0, 2 * np.pi, self.N + 1)[:-1]
        xx = radius * np.cos(phase_vec)
        yy = radius * np.sin(phase_vec)
        heading_vec = wrap(phase_vec + np.pi)
        vv = np.ones_like(phase_vec) * 0.4

        xx_ref = radius * np.cos(phase_vec + np.pi)
        yy_ref = radius * np.sin(phase_vec + np.pi)

        self.J_x_ref_fun = lambda i: np.array(
            [xx_ref[i], yy_ref[i], heading_vec[i], vv[i]])
        self.x_ref = np.vstack([self.J_x_ref_fun(i) for i in range(self.N)])
        assert self.x_ref.shape == (self.N, self.n)
        """ Traget Reference state"""
        self.J_Qr = np.eye(self.n) * 1.0
        """ Reference cost """
        self.J_Q_col = 20.0
        """ Collision cost """
        self.J_Q = np.eye(self.n) * 0.1
        """ State cost """
        self.J_R = np.eye(self.m) * 0.1
        """ Control cost """

        # Initial states, [N, n]
        self.x0 = np.vstack([xx, yy, heading_vec, vv]).T
        noise = np.random.uniform(-0.1, 0.1, size=(self.N, 2))
        self.x0[:, 0] += noise[:, 0]
        self.x0[:, 1] += noise[:, 1]
        self.guess = np.zeros((self.T, self.N, self.m))
        self.validate()

    def setup(self):
        # subclass responsible for loading cpp/eigen module
        # and setting x0
        if (self.config.USE_CPP or self.config.CPP_DEBUG or _CPP_DEBUG):
            self.cpp = cpp_Unicycle(
                self.N, self.T, self.dt, self.rho, self.rho_b, self.bc_a,
                self.bc_b, self.config.tolerance, self.backtracking_max_iter,
                self.J_Qr, self.J_Q, self.J_R, self.J_Q_col, self.x_ref,
                self.collision_diameter, self.config.iterations, False)
            self.cpp.set_x0(self.x0)

    def _visualize(self, u: np.ndarray, x: np.ndarray | None = None):
        """ Visualize state trajectory given experiment type
        Args:
            u: control trajectory, [T, N, m]
            x: state trajectory starting from to,
              [Any, N, n], if None, will be rollout from u
        Return:
            plotted fig object
        """
        if (x is None):
            x = np.vstack(
                [self.x0[np.newaxis, :, :],
                 self.rollout(self.x0, u)])
        fig, ax = plt.subplots()
        #colors = cm.get_cmap('viridis')(np.linspace(0, 1, self.N))
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        for i in range(self.N):
            # (x,y,heading,v)
            xx = x[:, i, 0]
            yy = x[:, i, 1]
            plt.plot(xx, yy, '-', color=colors[i])

        # plot initial and goal pose
        for i in range(self.N):
            center = self.x0[i, :2]
            # initial
            circle = patches.Circle(center,
                                    radius=0.1,
                                    fill=False,
                                    color=colors[i],
                                    linewidth=1)
            center = self.J_x_ref_fun(i)[:2]
            ax.add_patch(circle)
            circle = patches.Circle(center,
                                    radius=0.15,
                                    fill=False,
                                    color=colors[i],
                                    linewidth=2)
            ax.add_patch(circle)

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
        def collision_cost(i, j):
            dx = x_k[i, 0] - x_k[j, 0]
            dy = x_k[i, 1] - x_k[j, 1]
            return max(0, self.collision_diameter**2 - dx**2 - dy**2)

        cost += self.J_Q_col * sum(
            collision_cost(i, j) for j in range(self.N) if j != i)
        if (_CPP_DEBUG):
            alt = self.cpp.J(x_k, u_k_i, i)
            if (np.linalg.norm(alt - cost) > 1e-4):
                breakpoint()
        return cost

    def dJi_dxi(self, x_k: np.ndarray, u_k_i: np.ndarray,
                i: int) -> np.ndarray:
        """ Step cost gradient w.r.t. x_i
        Args:
            x_k: [N, n] *all* agent state at this step (k), (x, y, heading, v)
            u_k_i: [m] control for agent i at this step (k), (a, omega)
                a=dv_dt is acceleration
                omega=dheading_dt is angular acceleration
            i: agent id, starts from 0
            j: agent id, starts from 0
        Return:
            [n] Partial derivative
        """
        dx = x_k[i] - self.J_x_ref_fun(i)
        val = 2 * dx.T @ self.J_Qr + 2 * x_k[i].T @ self.J_Q

        def collision_cost_grad(i, j):
            """ Derivative w.r.t. x_i"""
            dx = x_k[i, 0] - x_k[j, 0]
            dy = x_k[i, 1] - x_k[j, 1]
            cost = self.collision_diameter**2 - dx**2 - dy**2
            return np.array([-2 * dx, -2 * dy, 0, 0]) if cost > 0 else 0

        for j in range(self.N):
            if i == j:
                continue
            val += self.J_Q_col * collision_cost_grad(i, j)
        assert val.shape == (self.n, )
        if (self.config.DEBUG or _DEBUG):

            def _J(x_i, u_k_i, i):
                _x_k = x_k.copy()
                _x_k[i] = x_i
                return self.J(_x_k, u_k_i, i)

            num = jacobianNumerical(lambda x_i: _J(x_i, u_k_i, i), x_k[i])
            assert (np.linalg.norm(num - val) < 1e-4)
        if (_CPP_DEBUG):
            alt = self.cpp.dJi_dxi(x_k, u_k_i, i)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    def dJi_dxj(self, x_k: np.ndarray, u_k_i: np.ndarray, i: int,
                j: int) -> np.ndarray:
        """ Step cost gradient w.r.t. x_j
        Args:
            x_k: [N, n] *all* agent state at this step (k), (x, y, heading, v)
            u_k_i: [m] control for agent i at this step (k), (a, omega)
                a=dv_dt is acceleration
                omega=dheading_dt is angular acceleration
            i: agent id, starts from 0
            j: agent id, starts from 0
        Return:
            [n] Partial derivative
        """
        val = np.zeros((1, self.n))
        d = self.collision_diameter

        def collision_cost_grad(i, j):
            dx = x_k[i, 0] - x_k[j, 0]
            dy = x_k[i, 1] - x_k[j, 1]
            cost = d**2 - dx**2 - dy**2
            return np.array([[2 * dx, 2 * dy, 0, 0]]) if cost > 0 else 0

        for j in range(self.N):
            if i == j:
                continue
            val += self.J_Q_col * collision_cost_grad(i, j)
        assert val.shape == (self.n, )
        if (self.config.DEBUG or _DEBUG):

            def _J(x_j, u_k_i, i, j):
                _x_k = x_k.copy()
                _x_k[j] = x_j
                return self.J(_x_k, u_k_i, i)

            num = jacobianNumerical(lambda x_j: _J(x_j, u_k_i, i, j), x_k[j])
            assert (np.linalg.norm(num - val) < 1e-4)
        if (_CPP_DEBUG):
            alt = self.cpp.dJi_dxj(x_k, u_k_i, i, j)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    def dJi_du(self, x_k: np.ndarray, u_k_i: np.ndarray, i: int):
        val = 2 * u_k_i.T @ self.J_R
        if (_CPP_DEBUG):
            alt = self.cpp.dJi_du(x_k, u_k_i, i)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    def dJi_dxi_dxi(self, x_k: np.ndarray, u_k_i: np.ndarray,
                    i: int) -> np.ndarray:
        """ Step cost second order derivative w.r.t. x_i
        Args:
            x_k: [N, n] *all* agent state at this step (k), (x, y, heading, v)
            u_k_i: [m] control for agent i at this step (k), (a, omega)
                a=dv_dt is acceleration
                omega=dheading_dt is angular acceleration
            i: agent id, starts from 0
            j: agent id, starts from 0
        Return:
            [n, n] Partial derivative
        """
        val = 2 * self.J_Qr + 2 * self.J_Q

        def collision_cost_hess(i, j):
            dx = x_k[i, 0] - x_k[j, 0]
            dy = x_k[i, 1] - x_k[j, 1]
            cost = self.collision_diameter**2 - dx**2 - dy**2
            return np.diag([2, 2, 0, 0]) if cost > 0 else 0

        for j in range(self.N):
            if i == j:
                continue
            val += collision_cost_hess(i, j)
        assert val.shape == (self.n, self.n)
        if (_CPP_DEBUG):
            alt = self.cpp.dJi_dxi_dxi(x_k, u_k_i, i)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    def dJi_dxi_dxj(self, x_k: np.ndarray, u_k_i: np.ndarray, i: int,
                    j: int) -> np.ndarray:
        """ Step cost second ordder derivative w.r.t. x_i, then x_j
        Args:
            x_k: [N, n] *all* agent state at this step (k), (x, y, heading, v)
            u_k_i: [m] control for agent i at this step (k), (a, omega)
                a=dv_dt is acceleration
                omega=dheading_dt is angular acceleration
            i: agent id, starts from 0
            j: agent id, starts from 0
        Return:
            [n, n] Partial derivative
        """
        return np.zeros((self.n, self.n))

    def dJi_dxj_dxj(self, x_k: np.ndarray, u_k_i: np.ndarray, i: int,
                    j: int) -> np.ndarray:
        """ Step cost second order derivative w.r.t. x_j
        Args:
            x_k: [N, n] *all* agent state at this step (k), (x, y, heading, v)
            u_k_i: [m] control for agent i at this step (k), (a, omega)
                a=dv_dt is acceleration
                omega=dheading_dt is angular acceleration
            i: agent id, starts from 0
            j: agent id, starts from 0
        Return:
            [n, n] Partial derivative
        """
        val = 2 * self.J_Qr + 2 * self.J_Q

        def collision_cost_hess(i, j):
            dx = x_k[i, 0] - x_k[j, 0]
            dy = x_k[i, 1] - x_k[j, 1]
            cost = self.collision_diameter**2 - dx**2 - dy**2
            return np.diag([2, 2, 0, 0]) if cost > 0 else 0

        for j in range(self.N):
            if i == j:
                continue
            val += collision_cost_hess(i, j)
        if (_CPP_DEBUG):
            alt = self.cpp.dJi_dxj_dxj(x_k, u_k_i, i, j)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    def dJi_dudu(self, x_k, u_k_i, i):
        val = 2 * self.J_R
        if (_CPP_DEBUG):
            alt = self.cpp.dJi_dudu(x_k, u_k_i, i)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

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
        val = x + dxdt * self.dt
        assert val.shape == (self.n, )
        if (_CPP_DEBUG):
            alt = self.cpp.f(x, u, i)
            if (np.linalg.norm(alt.flatten() - val) > 1e-4):
                breakpoint()
        return val

    def df_dx(self, x: np.ndarray, u: np.ndarray, i: int) -> np.ndarray:
        """ Dynamics derivative
        Args:
            x: [n] states of agent i
                x = (x, y, heading, v)
            u: [m] control of agent i
                u = (a, omega)
            i: agent index 
        Return:
            [n,n] Derivative
        """
        x0, x1, x2, x3 = x
        u0, u1 = u
        dfdx = np.array([[0, 0, -x3 * sin(x2), cos(x2)],
                         [0, 0, x3 * cos(x2), sin(x2)], [0, 0, 0, 0],
                         [0, 0, 0, 0]])
        val = np.eye(4) + dfdx * self.dt
        assert val.shape == (self.n, self.n)
        if (self.config.DEBUG or _DEBUG):
            num = jacobianNumerical(lambda xx: self.f(xx, u, i), x, dim=self.n)
            assert (np.linalg.norm(num - val) < 1e-4)
        if (_CPP_DEBUG):
            alt = self.cpp.df_dx(x, u, i)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    def df_du(self, x: np.ndarray, u: np.ndarray, i: int) -> np.ndarray:
        """
        Args:
            x: [n] states of agent i
                x = (x, y, heading, v)
            u: [m] control of agent i
                u = (a, omega)
            i: agent index 
        Return:
            [n,m] Derivative
        """
        dfdu = np.array([[0, 0], [0, 0], [0, 1], [1, 0]])
        val = dfdu * self.dt
        assert val.shape == (self.n, self.m)
        if (self.config.DEBUG or _DEBUG):
            num = jacobianNumerical(lambda uu: self.f(x, uu, i), u, dim=self.n)
            assert (np.linalg.norm(num - val) < 1e-4)
        if (_CPP_DEBUG):
            alt = self.cpp.df_du(x, u, i)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

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

    def check_collision(self, x: np.ndarray):
        """ Verify soft collision constraint is met"""
        cost = 0
        min_dist_between_agents = np.inf
        d = self.collision_diameter
        for i, j in itertools.combinations(range(self.N), 2):
            dx = x[:, i, 0] - x[:, j, 0]
            dy = x[:, i, 1] - x[:, j, 1]
            min_dist_sqr = np.min(dx**2 + dy**2)
            if (min_dist_sqr < min_dist_between_agents**2):
                min_dist_between_agents = min_dist_sqr**0.5
            cost += self.J_Q_col * np.sum(
                np.clip(d**2 - dx**2 - dy**2, 0, None))
        print(f'{cost=}')
        print(f'{min_dist_between_agents=}')


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{:7.3f}'.format})
    _config = ResidualGameConfig(tolerance=1e-2,
                                 iterations=50,
                                 DEBUG=False,
                                 USE_CPP=True,
                                 CPP_DEBUG=False)
    print(_config)
    main = Unicycle(_config, agent_count=3)
    #main.get_symbolic_dynamics()
    main.setup()
    logger.info('testing')
    main.visualize(main.guess, save_fig=True, fig_name='initial')
    u_ref, full_x_ref, has_converged = main.solve()
    main.final()
    main.visualize(u_ref, save_fig=True, fig_name='solution')
    main.check_collision(full_x_ref)
