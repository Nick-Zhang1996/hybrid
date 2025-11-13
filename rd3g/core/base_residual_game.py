from abc import abstractmethod

import numpy as np
import jax.numpy as jnp

from rd3g.core.base_game import BaseGame


class BaseResidualGame(BaseGame):
    """ Defines additional functions a game needs to provide for RD3G (without autograd) to work"""

    # ---------- Defaults for  some Application specific functions -------
    # terminal(final) cost for agent i
    # x_T: terminal GAME state (N*n)
    # return : scalar
    # if User doesn't choose a terminal cost, the step cost J will be used

    def Jfi(self, x_T, i):
        return self.J(x_T, np.zeros(self.m), i)

    def jax_Jfi(self, x_T, i):
        return self.jax_J(x_T, jnp.zeros(self.m), i)

    def dJfi_dxi(self, x_T, i):
        return self.dJi_dxi(x_T, np.zeros(self.m), i)

    def dJfi_dxj(self, x_T, i, j):
        return self.dJi_dxj(x_T, np.zeros(self.m), i, j)

    def dJfi_dxi_dxi(self, x_T, i):
        return self.dJi_dxi_dxi(x_T, np.zeros(self.m), i)

    def dJfi_dxi_dxj(self, x_T, i, j):
        return self.dJi_dxi_dxj(x_T, np.zeros(self.m), i, j)

    def dJfi_dxj_dxj(self, x_T, i, j):
        return self.dJi_dxj_dxj(x_T, np.zeros(self.m), i, j)

    # step cost function
    @abstractmethod
    def J(self, x_k: np.ndarray, u_k_i: np.ndarray, i: int) -> float:
        """Step cost function for agent i.

        Args:
            x_k: [N, n] *all* agent state at this step (k)
            u_k_i: [m] control for agent i at this step (k)
            i: agent id, starts from 0
        Return:
            cost for agent i at this step (k)

        """
        raise NotImplementedError

    @abstractmethod
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
        return np.zeros((self.n))

    @abstractmethod
    def dJi_dxj(self, x_k: np.ndarray, u_k_i: np.ndarray, i: int,
                j: int) -> np.ndarray:
        """ Step cost gradient w.r.t. x_j
        Args:
            x_k: [N, n] *all* agent state at this step (k)
            u_k_i: [m] control for agent i at this step (k)
            i: agent id, starts from 0
            j: agent id, starts from 0
        Return:
            [n] Partial derivative
        """
        return np.zeros((self.n))

    @abstractmethod
    def dJi_dxi_dxi(self, x_k: np.ndarray, u_k_i: np.ndarray, i: int):
        """ Step cost second order derivative w.r.t. x_i
        Args:
            x_k: [N, n] *all* agent state at this step (k)
            u_k_i: [m] control for agent i at this step (k)
            i: agent id, starts from 0
            j: agent id, starts from 0
        Return:
            [n, n] Partial derivative
        """
        return np.zeros((self.n, self.n))

    @abstractmethod
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

    @abstractmethod
    def dJi_dxj_dxj(self, x_k: np.ndarray, u_k_i: np.ndarray, i: int,
                    j: int) -> np.ndarray:
        """ Step cost second order derivative w.r.t. x_j
        Args:
            x_k: [N, n] *all* agent state at this step (k)
            u_k_i: [m] control for agent i at this step (k)
            i: agent id, starts from 0
            j: agent id, starts from 0
        Return:
            [n, n] Partial derivative
        """
        return np.zeros((self.n, self.n))

    @abstractmethod
    def dJi_du(self, x_k, u_k_i, i):
        return np.zeros((1, self.m))

    @abstractmethod
    def dJi_dudu(self, x_k, u_k_i, i):
        return np.zeros((self.m, self.m))

    # --- dynamics and related derivatives ---
    @abstractmethod
    def f(self, x: np.ndarray, u: np.ndarray, i: int) -> np.ndarray:
        """ Dynamics funciton, gives x(state) at next time step 
        Args:
            x: [n] states of agent i
            u: [m] control of agent i
            i: agent index 
        Return:
           States [n] at next time step

        """
        raise NotImplementedError

    @abstractmethod
    def df_dx(self, x: np.ndarray, u: np.ndarray, i: int) -> np.ndarray:
        """ Dynamics derivative df/dx
        Args:
            x: [n] states of agent i
            u: [m] control of agent i
            i: agent index 
        Return:
            [n,n] State derivative
        """
        del x
        del u
        del i
        raise NotImplementedError

    @abstractmethod
    def df_du(self, x: np.ndarray, u: np.ndarray, i: int) -> np.ndarray:
        """
        Derivative df/du
        Args:
            x: [n] states of agent i
                x = (x, y, heading, v)
            u: [m] control of agent i
                u = (a, omega)
            i: agent index 
        Return:
            [n,m] Derivative
        """
        del x
        del u
        del i
        raise NotImplementedError

    def h(self, x_i: jnp.ndarray, x_j: jnp.ndarray) -> float:
        """ Constraint function h <= 0"""
        del x_i
        del x_j
        return -1

    def dh_dxi(self, x_i: jnp.ndarray, x_j: jnp.ndarray) -> jnp.ndarray:
        """ Derivative of constraint h, dh/dx_i
        Return:
            [n] Derivative
        """
        del x_i
        del x_j
        return jnp.zeros((self.n))

    def dh_dxj(self, x_i: jnp.ndarray, x_j: jnp.ndarray) -> jnp.ndarray:
        """ Derivative of constraint h, dh/dx_j
        Return:
            [n] Derivative
        """
        del x_i
        del x_j
        return jnp.zeros((self.n))

    def dh_dxi_dxi(self, x_i, x_j):
        del x_i
        del x_j
        return jnp.zeros((self.n, self.n))

    def dh_dxj_dxi(self, x_i, x_j):
        del x_i
        del x_j
        return jnp.zeros((self.n, self.n))

    def dh_dxi_dxj(self, x_i, x_j):
        del x_i
        del x_j
        return np.zeros((self.n, self.n))

    def dh_dxj_dxj(self, x_i, x_j):
        del x_i
        del x_j
        return np.zeros((self.n, self.n))
