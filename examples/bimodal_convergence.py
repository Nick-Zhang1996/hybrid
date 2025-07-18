import numpy as np
import matplotlib.pyplot as plt
import pickle as p

from utilities.util import *
from utilities.time_util import TimeUtil

from residual_game import ResidualGame, ResidualGameConfig
from stein_game import SteinGame, SteinGameConfig


# example for paper Stein Variational Game, Low Dimension Examples 1)
class BimodalConvergence(SteinGame):

    def __init__(self, config: SteinGameConfig):
        super().__init__(config)
        '''
        x_i_k: 1..T, T*N*n  NOTE starts from 1
        u_i_k: 0..T-1, T*N*m
        lamda_i_k: 0..T-1 T*N*n
        mu_k_i_j: 1..T T*N*N NOTE starts from 1
        '''
        #self.print_debug_enable()
        self.N = 2
        self.T = 5
        self.dt = dt = 0.2
        self.dynamics_residual_weight = 1.0

        self.config.particles = 100

        self.config.stein_iterations = 10  # 30

        # x^i: [position, velocity (1D)]
        # u^i: [acceleration]
        self.n = 2
        self.m = 1
        # cost tuning parameter
        self.C1 = 2
        self.C2 = 1

        # stein sampling prior
        self.dim_theta = self.T * self.N * self.m
        self.covariance_mtx = np.diag([8] * 2)

        # bounds for visualization
        self.visual_x_lim = [-2.5, 2.5]
        self.visual_y_lim = [-2.5, 2.5]

        # step cost parameters
        # NOTE this lambda fun needs to be implemented in c++
        self.J_Q = np.diag([1, 0])
        self.J_R = np.eye(self.m) * 1e-2
        self.p1 = np.array([-1, 1])
        self.p2 = np.array([1, -1])
        self.A = np.array([[1, dt], [0, 1]])
        self.B = np.array([[0.5 * dt * dt], [dt]])

        #self.x0 = np.array([[1,0],[-1,0]])
        self.x0 = np.zeros((self.N, self.n))

        # initial guess for u
        self.guess = np.zeros((self.T, self.N, self.m))
        self.guess[:, 0, 0] = -2.50
        self.guess[:, 1, 0] = 5.12

    def setup(self):
        # subclass responsible for loading cpp/eigen module
        if (self.config.USE_CPP or self.config.CPP_DEBUG):
            self.print_error('cpp implementation not available')

    # use a uniform prior
    def initialSample(self, count=None):
        ''' make [self.config.particles] samples of size theta from an initial belief'''
        if (count is None):
            count = self.config.particles
        val = np.random.multivariate_normal(np.zeros(2), self.covariance_mtx,
                                            count)[:, :, np.newaxis]
        #val = np.random.uniform(-self.covariance_mtx[0,0],self.covariance_mtx[0,0], count*2).reshape(-1,2)
        val = np.tile(val, (1, self.T, 1))
        dim_u = (self.T, self.N, self.m)
        return val.reshape(count, -1)

    def _visualize(self, u, x=None):
        if (x is None):
            x = np.vstack(
                [self.x0[np.newaxis, :, :],
                 self.rollout(self.x0, u)])
        fig, ax = plt.subplots()
        # target points, p1, p2
        ax.plot([-1], [1], 'o')
        ax.plot([1], [-1], 'o')

        # plot agent 0,1's position as x,y coordinate
        xx = x[:, 0, 0]
        yy = x[:, 1, 0]
        plt.plot(xx, yy, '*-')
        ax.set_aspect('equal', adjustable='box')
        return fig

    def _animation(self, u, x=None, gif_prefix=''):
        return None

    ''' --------  math functions and their derivatives ------ '''

    # step cost
    def J(self, x_k, u_k_i, i):
        '''
        step cost for an agent, given x,u
        x_k.shape (N*n) x_k_i = [r,v] : pos, vel
        u_k_i.shape (m) u_k_i = [a] : acc
        i: agent id
        '''
        val = u_k_i.T @ self.J_R @ u_k_i
        return val + self.Jfi(x_k, i)

    def dJi_dxi(self, x_k, u_k_i, i):
        return np.zeros((1, self.n)) + self.dJfi_dxi(x_k, i)

    def dJi_dxj(self, x_k, u_k_i, i, j):
        return np.zeros((1, self.n)) + self.dJfi_dxj(x_k, i, j)

    def dJi_dxi_dxi(self, x_k, u_k_i, i):
        return np.zeros((self.n, self.n)) + self.dJfi_dxi_dxi(x_k, i)

    def dJi_dxi_dxj(self, x_k, u_k_i, i, j):
        return np.zeros((self.n, self.n)) + self.dJfi_dxi_dxj(x_k, i, j)

    def dJi_dxj_dxj(self, x_k, u_k_i, i, j):
        return np.zeros((self.n, self.n)) + self.dJfi_dxj_dxj(x_k, i, j)

    def dJi_du(self, x_k, u_k_i, i):
        val = 2 * u_k_i.T @ self.J_R
        return val

    def dJi_dudu(self, x_k, u_k_i, i):
        return 2 * self.J_R

    # terminal cost
    def Jfi(self, x_T, i):
        f1 = (x_T[0] - self.p1).T @ self.J_Q @ (x_T[0] - self.p1)  \
             +  (x_T[1] - self.p2).T @ self.J_Q @ (x_T[1] - self.p2)
        f2 = (x_T[0] - self.p2).T @ self.J_Q @ (x_T[0] - self.p2)  \
             +  (x_T[1] - self.p1).T @ self.J_Q @ (x_T[1] - self.p1)
        return -np.exp(-f1) - np.exp(-f2)

    def dJfi_dxi(self, x_T, i):
        # J = J1 + J2 = [- np.exp(-f1)] + [- np.exp(-f2)]
        f1 = (x_T[0] - self.p1).T @ self.J_Q @ (x_T[0] - self.p1)  \
             +  (x_T[1] - self.p2).T @ self.J_Q @ (x_T[1] - self.p2)
        f2 = (x_T[0] - self.p2).T @ self.J_Q @ (x_T[0] - self.p2)  \
             +  (x_T[1] - self.p1).T @ self.J_Q @ (x_T[1] - self.p1)
        if (i == 0):
            df1_dx0 = 2 * (x_T[0] - self.p1).T @ self.J_Q
            df1_dx0dx0 = 2 * self.J_Q
            df2_dx0 = 2 * (x_T[0] - self.p2).T @ self.J_Q
            df2_dx0dx0 = 2 * self.J_Q
            dJ1_dx0 = np.exp(-f1) * df1_dx0
            dJ2_dx0 = np.exp(-f2) * df2_dx0
            retval = dJ1_dx0 + dJ2_dx0
        else:
            df1_dx0 = 2 * (x_T[1] - self.p2).T @ self.J_Q
            df1_dx0dx0 = 2 * self.J_Q
            df2_dx0 = 2 * (x_T[1] - self.p1).T @ self.J_Q
            df2_dx0dx0 = 2 * self.J_Q
            dJ1_dx0 = np.exp(-f1) * df1_dx0
            dJ2_dx0 = np.exp(-f2) * df2_dx0
            retval = dJ1_dx0 + dJ2_dx0

        if (self.config.DEBUG):
            dJfi_dxi_num = jacobianNumerical(
                lambda xx: self.Jfi(xx.reshape(x_T.shape), i),
                x_T.flatten(),
                dim=1)
            dJfi_dxi_num = dJfi_dxi_num[:, i * 2:(i + 1) * 2]

            self.print_debug(
                f'dJfi_dxi err {np.linalg.norm(dJfi_dxi_num - retval)}')
            assert (np.linalg.norm(dJfi_dxi_num - retval) < 1e-4)
        return retval

    def dJfi_dxj(self, x_T, i, j):
        return self.dJfi_dxi(x_T, j)

    def dJfi_dxi_dxi(self, x_T, i):
        Q = self.J_Q
        f1 = (x_T[0] - self.p1).T @ self.J_Q @ (x_T[0] - self.p1)  \
             +  (x_T[1] - self.p2).T @ self.J_Q @ (x_T[1] - self.p2)
        f2 = (x_T[0] - self.p2).T @ self.J_Q @ (x_T[0] - self.p2)  \
             +  (x_T[1] - self.p1).T @ self.J_Q @ (x_T[1] - self.p1)
        if (i == 0):
            df1_dx0 = 2 * (x_T[0] - self.p1).reshape(1, -1) @ self.J_Q
            df1_dx0dx0 = 2 * self.J_Q
            df2_dx0 = 2 * (x_T[0] - self.p2).reshape(1, -1) @ self.J_Q
            df2_dx0dx0 = 2 * self.J_Q
            retval = df1_dx0.T * np.exp(-f1) @ -df1_dx0  + np.exp(-f1) * df1_dx0dx0 \
                    + df2_dx0.T * np.exp(-f2) @ -df2_dx0  + np.exp(-f2) * df2_dx0dx0
        else:
            df1_dx0 = 2 * (x_T[1] - self.p2).reshape(1, -1) @ self.J_Q
            df1_dx0dx0 = 2 * self.J_Q
            df2_dx0 = 2 * (x_T[1] - self.p1).reshape(1, -1) @ self.J_Q
            df2_dx0dx0 = 2 * self.J_Q
            retval = df1_dx0.T * np.exp(-f1)  @ -df1_dx0  + np.exp(-f1) * df1_dx0dx0 \
                    + df2_dx0.T * np.exp(-f2)  @ -df2_dx0  + np.exp(-f2) * df2_dx0dx0

        if (self.config.DEBUG):
            dJfi_dxi_dxi_num = jacobianNumerical(
                lambda xx: self.dJfi_dxi(xx.reshape(x_T.shape), i),
                x_T.flatten(),
                dim=self.n)
            dJfi_dxi_dxi_num = dJfi_dxi_dxi_num[:, i * 2:(i + 1) * 2]
            self.print_debug(
                f'dJfi_dxi_dxi err {np.linalg.norm(dJfi_dxi_dxi_num - retval)}'
            )
            assert (np.linalg.norm(dJfi_dxi_dxi_num - retval) < 1e-4)
        return retval

    def dJfi_dxi_dxj(self, x_T, i, j):
        Q = self.J_Q
        f1 = (x_T[0] - self.p1).T @ self.J_Q @ (x_T[0] - self.p1)  \
             +  (x_T[1] - self.p2).T @ self.J_Q @ (x_T[1] - self.p2)
        f2 = (x_T[0] - self.p2).T @ self.J_Q @ (x_T[0] - self.p2)  \
             +  (x_T[1] - self.p1).T @ self.J_Q @ (x_T[1] - self.p1)
        assert (i != j)
        if (i == 0):
            df1_dx0 = 2 * (x_T[0] - self.p1).reshape(1, -1) @ self.J_Q
            df1_dx1 = 2 * (x_T[1] - self.p2).reshape(1, -1) @ self.J_Q
            df1_dx0dx0 = 2 * self.J_Q
            df2_dx0 = 2 * (x_T[0] - self.p2).reshape(1, -1) @ self.J_Q
            df2_dx1 = 2 * (x_T[1] - self.p1).reshape(1, -1) @ self.J_Q
            df2_dx0dx0 = 2 * self.J_Q
            dJ1_dx0 = f1 * np.exp(-f1) * df1_dx0
            retval = df1_dx0.T * np.exp(-f1)  @ -df1_dx1  \
                    + df2_dx0.T * np.exp(-f2) @ -df2_dx1
        else:
            # here x0 = xi = x1, a bit confusing
            df1_dx0 = 2 * (x_T[1] - self.p2).reshape(1, -1) @ self.J_Q
            df1_dx1 = 2 * (x_T[0] - self.p1).reshape(1, -1) @ self.J_Q
            df1_dx0dx0 = 2 * self.J_Q
            df2_dx0 = 2 * (x_T[1] - self.p1).reshape(1, -1) @ self.J_Q
            df2_dx1 = 2 * (x_T[0] - self.p2).reshape(1, -1) @ self.J_Q
            df2_dx0dx0 = 2 * self.J_Q
            dJ1_dx0 = f1 * np.exp(-f1) * df1_dx0
            retval = df1_dx0.T * np.exp(-f1)  @ -df1_dx1  \
                    + df2_dx0.T * np.exp(-f2) @ -df2_dx1

        if (self.config.DEBUG):
            dJfi_dxi_dxj_num = jacobianNumerical(
                lambda xx: self.dJfi_dxi(xx.reshape(x_T.shape), i),
                x_T.flatten(),
                dim=self.n)
            dJfi_dxi_dxj_num = dJfi_dxi_dxj_num[:, j * 2:(j + 1) * 2]
            self.print_debug(
                f'dJfi_dxi_dxj err {np.linalg.norm(dJfi_dxi_dxj_num - retval)}'
            )
            assert (np.linalg.norm(dJfi_dxi_dxj_num - retval) < 1e-4)
        return retval

    def dJfi_dxj_dxj(self, x_T, i, j):
        return self.dJfi_dxi_dxi(x_T, j)

    # --- dynamics and related derivatives ---
    # this problem has homogeneous agents, so [i] is irrelevant
    def f(self, x, u, i):
        return self.A @ x + self.B @ u

    def df_dx(self, x, u, i):
        return self.A

    def df_du(self, x, u, i):
        return self.B

    def h(self, x_i, x_j):
        return -1

    def dh_dxi(self, x_i, x_j):
        return np.zeros((1, self.n))

    def dh_dxj(self, x_i, x_j):
        return np.zeros((1, self.n))

    def dh_dxi_dxi(self, x_i, x_j):
        return np.zeros((self.n, self.n))

    def dh_dxj_dxi(self, x_i, x_j):
        return np.zeros((self.n, self.n))

    def dh_dxi_dxj(self, x_i, x_j):
        return np.zeros((self.n, self.n))

    def dh_dxj_dxj(self, x_i, x_j):
        return np.zeros((self.n, self.n))

    def testAnimation(self):
        u_ref = np.zeros((self.T, self.N, self.m))
        u_ref[:, 0, :] = 1
        u_ref[:, 1, :] = -2
        x_ref = self.rollout(self.x0, u_ref)
        full_x_ref = np.vstack([self.x0[np.newaxis, :, :], x_ref])
        self._visualize(u_ref, full_x_ref)
        plt.show()

    def getAgentCost(self, x, u):
        cost_vec = []
        for i in range(self.N):
            cost = 0
            for k in range(self.T):
                cost += self.J(x[k], u[k, i], i)

            final_cost = self.Jfi(x[self.T], i)
            cost += final_cost
            print(f'agent {i}, total cost {cost} final cost {final_cost}')
            cost_vec.append(cost)
        return cost_vec


if __name__ == "__main__":
    #np.random.seed(0)
    main = BimodalConvergence(SteinGameConfig())
    main.setup()
    u_ref, full_x_ref, has_converged = main.solve()
    main.final()
    print(
        f'u_ref mean {np.mean(u_ref.flatten())} std {np.std(u_ref.flatten())}, has_converged: {has_converged}'
    )
    cost_vec = main.getAgentCost(full_x_ref, u_ref)
    print(f'cost_vec: {cost_vec}')

    #self.belief_support = np.array(good_u_ref)
    #self.belief_weight = np.array([1/len(good_u_ref)]*len(good_u_ref))
    # this is actually the residual, we use that as cost in the Stein descent
    #self.belief_support_residual = np.array(good_cost_ref)
    # total agent cost, this is the negative social utility
    #self.belief_support_cost = np.array(good_agent_cost)

    # plot end position
    x_ref_vec = []
    for u_ref in main.belief_support:
        x_ref = main.rollout(main.x0, u_ref)
        x_ref_vec.append(x_ref)
    x_ref_vec = np.array(x_ref_vec)

    fig, ax = plt.subplots()
    # target points, p1, p2
    ax.plot([main.p1[0]], [main.p1[1]], 'o')
    ax.plot([main.p2[0]], [main.p2[1]], 'o')

    # plot agent 0,1's position as x,y coordinate
    xx = x_ref_vec[:, -1, 0, 0]
    yy = x_ref_vec[:, -1, 1, 0]
    plt.plot(xx, yy, '*')
    ax.set_aspect('equal', adjustable='box')
    plt.show()
    data = {
        'belief_support': main.belief_support,
        'belief_weight': main.belief_weight,
        'belief_support_residual': main.belief_support_residual,
        'belief_support_cost': main.belief_support_cost,
        'belief_x_ref': main.belief_x_ref,
        'particle_history': main.particle_history
    }
    with open('particles.p', 'wb') as f:
        p.dump(data, f)
    print(
        np.sum(np.linalg.norm(main.belief_x_ref[:, -1, :, 0], axis=1) > 0.5) /
        270)
    #main.testAnimation()
