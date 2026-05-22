"""Proof of concept on why preconditioning with potential is faster."""

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np


class Example:
    def __init__(self, step_size=0.05, num_steps=100, x0=(2.5, -1.3)):
        self.step_size = step_size
        self.num_steps = num_steps
        self.x0 = np.array(x0, dtype=float)

        x = cas.SX.sym("x")
        y = cas.SX.sym("y")
        z = cas.vertcat(x, y)

        val = 1.2
        j1 = x**2 + y**2 + 0.2*val * x**2 * y - 4*y
        j2 = 1.3*x**2 + 1.6*y**2 - 2*val * x * y - 3*x
        r = cas.vertcat(cas.gradient(j1, x), cas.gradient(j2, y))
        dr = cas.jacobian(r, z)

        s = 0.5 * (dr + dr.T)
        a = 0.5 * (dr - dr.T)

        self.j1_fn = cas.Function("J1", [z], [j1])
        self.j2_fn = cas.Function("J2", [z], [j2])
        self.r_fn = cas.Function("R", [z], [r])
        self.dr_fn = cas.Function("dR", [z], [dr])
        self.s_fn = cas.Function("S", [z], [s])
        self.a_fn = cas.Function("A", [z], [a])

    @staticmethod
    def _normalize(vec):
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            return np.zeros_like(vec)
        return vec / norm

    def newton_update(self, x, y):
        z = np.array([x, y], dtype=float)
        r = np.array(self.r_fn(z)).astype(float).reshape(-1)
        dr = np.array(self.dr_fn(z)).astype(float)
        dz = np.linalg.solve(dr, r)
        return dz

    def precondition_update(self, x, y, precond_iter):
        z = np.array([x, y], dtype=float)
        r = np.array(self.r_fn(z)).astype(float).reshape(-1)
        s = np.array(self.s_fn(z)).astype(float)
        a = np.array(self.a_fn(z)).astype(float)
        inv_sa = np.linalg.solve(s, a)
        eigvals = np.linalg.eigvals(inv_sa)
        print(eigvals)
        spectral_radius = np.max(np.abs(eigvals))
        print(f"rho(inv(S)A) = {spectral_radius:.6f}")
        principal_eig = eigvals[np.argmax(np.imag(eigvals))]
        rotation_deg = np.degrees(np.angle(principal_eig))
        print(f"inv(S)A radius = {np.abs(principal_eig):.6f}, rotation = {rotation_deg:.6f} deg")
        dz_last = np.zeros(2, dtype=float)
        for _ in range(precond_iter):
            rhs = r - a @ dz_last
            dz = np.linalg.solve(s, rhs)
            dz_last = dz
        return dz

    def _rollout_newton(self):
        z = self.x0.copy()
        path = [z.copy()]

        for _ in range(self.num_steps):
            dz = self.newton_update(z[0], z[1])
            step = self.step_size * self._normalize(dz)
            z = z - step
            path.append(z.copy())

        return np.array(path)

    def _rollout_precondition(self, precond_iter):
        z = self.x0.copy()
        # dz_last = np.zeros(2, dtype=float)
        path = [z.copy()]

        for _ in range(self.num_steps):
            dz = self.precondition_update(z[0], z[1], precond_iter)
            step = self.step_size * self._normalize(dz)
            z = z - step
            path.append(z.copy())
            # dz_last = dz

        return np.array(path)

    def _cost_grid(self, fn, xlim=(-3.0, 3.0), ylim=(-3.0, 3.0), grid_size=250):
        xs = np.linspace(xlim[0], xlim[1], grid_size)
        ys = np.linspace(ylim[0], ylim[1], grid_size)
        xx, yy = np.meshgrid(xs, ys)
        values = np.zeros_like(xx)

        for i in range(grid_size):
            for j in range(grid_size):
                values[i, j] = float(fn(np.array([xx[i, j], yy[i, j]], dtype=float)))

        return xx, yy, values

    def _plot_landscape(self, ax, fn, title, newton_path, precondition_paths):
        xx, yy, values = self._cost_grid(fn)
        contour = ax.contourf(xx, yy, values, levels=30, cmap="coolwarm")
        plt.colorbar(contour, ax=ax, shrink=0.9, label="Cost")

        ax.plot(
            newton_path[:, 0],
            newton_path[:, 1],
            color="white",
            linewidth=2.5,
            label="Newton-like update",
        )
        gradient = plt.cm.Reds(np.linspace(0.35, 0.95, len(precondition_paths)))
        for precond_iter, (precondition_path, color) in enumerate(zip(precondition_paths, gradient)):
            ax.plot(
                precondition_path[:, 0],
                precondition_path[:, 1],
                color=color,
                linewidth=2.0,
                linestyle="--",
                label="Preconditioned update" if precond_iter == 0 else None,
            )

        ax.scatter(*self.x0, color="black", s=40, label="Start")
        ax.scatter(0.0, 0.0, color="gold", edgecolor="black", s=60, label="Nash equilibrium")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.legend(loc="upper right")

    def run(self):
        newton_path = self._rollout_newton()
        precondition_paths = [self._rollout_precondition(
            precond_iter) for precond_iter in range(1, 10)]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        self._plot_landscape(axes[0], self.j1_fn,
                             "Player 1 Cost Landscape ($J_1$)", newton_path, precondition_paths)
        self._plot_landscape(axes[1], self.j2_fn,
                             "Player 2 Cost Landscape ($J_2$)", newton_path, precondition_paths)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    Example().run()
