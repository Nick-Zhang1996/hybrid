""" Base class for Jax game. This is OBSELETE, jax didn't work well with small, sparse problem.
Use CasadiGame instead """
from functools import partial
from jax.typing import ArrayLike
from jax.lax import scan
from jax import vmap, jit
import jax.numpy as jnp
from rd3g.core.base_game import BaseGame


class BaseJaxGame(BaseGame):
    """ Base game class for JAX compatibility"""

    @partial(jit, static_argnums=0)
    def rollout(self, x0: ArrayLike, u: ArrayLike) -> ArrayLike:
        """ Rollout control to get state trajectory
        Args:
            x0: (N,n)
            u: (T,N,m), u0..u_T-1, will be reshaped
        Return:
            X: (T,N,n) x1..xT
        """
        assert u.shape == (self.T, self.N, self.m)
        X = jnp.zeros((self.T + 1, self.N, self.n))

        def _get_next_state(x_k_i, u_k_i, i):
            next_state = self.f(x_k_i, u_k_i, i).flatten()
            return next_state

        get_next_state_all_agents = vmap(_get_next_state, in_axes=(0, 0, 0))

        def scan_fun(states, u_k):  # carry, val
            next_states = get_next_state_all_agents(states, u_k, jnp.arange(self.N))
            return next_states, next_states
        _, state_traj = scan(scan_fun, x0.reshape(self.N, self.n), u)
        return state_traj
