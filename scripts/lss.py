# Illustration for Local Symplectic Surgery
import numpy as np
import casadi as cas
import matplotlib.pyplot as plt


class LSS:
    """ Use local symplectic surgery to solve a demo zero-sum game problem"""

    def __init__(self):
        # 1. Define symbolic variables
        self.x = cas.SX.sym('x', 1, 1)
        self.y = cas.SX.sym('y', 1, 1)

        # 2. Define the exact cost function J
        self.J_val = self.J(self.x, self.y)

        # 3. Compute gradients for the game vector field w
        Jx = cas.jacobian(self.J_val, self.x)
        Jy = cas.jacobian(self.J_val, self.y)

        # Player 1 minimizes J, Player 2 maximizes J (minimizes -J)
        w = cas.vertcat(Jx, -Jy)
        self.z = z = cas.vertcat(self.x, self.y)

        # 4. Compute Jacobian of the vector field
        self.Dw = Dw = cas.jacobian(w, self.z)

        # 5. Define Continuous-Time Dynamics
        I = np.eye(2)

        # Gradient Descent follows the negative vector field
        self.dzdt_gd = -w

        # Local Symplectic Surgery corrects the field
        lamda = 1e-4 * (1 - cas.exp(-cas.norm_2(w)**2))
        v = Dw.T @ cas.inv(Dw.T @ Dw + lamda*I) @ Dw.T @ w
        self.dzdt_lss = -(w + cas.exp(-1e-4 * cas.norm_2(v)**2) * v)

        Jxx = self.Dw[0, 0]
        Jxy = self.Dw[0, 1]
        Jyx = -self.Dw[1, 0]
        Jyy = -self.Dw[1, 1]

        # Regularization to prevent division by zero in flat regions during RK4 integration
        eps = 1e-8
        dx_dftr = -Jx - (1.0 / (Jxx + eps)) * Jxy * Jy
        dy_dftr = Jy + (1.0 / (Jyy + eps)) * Jyx * Jx
        self.dzdt_dftr = cas.vertcat(dx_dftr, dy_dftr)

        # 6. Create fast CasADi functions for the integrators
        self.f_gd = cas.Function('f_gd', [z], [self.dzdt_gd])
        self.f_lss = cas.Function('f_lss', [z], [self.dzdt_lss])
        self.f_dftr = cas.Function('f_dftr', [z], [self.dzdt_dftr])

        # Evaluator for Dw
        self.eval_Dw = cas.Function('eval_Dw', [z], [Dw])
        self.eval_w = cas.Function('eval_w', [z], [w])

        # CasADi Rootfinder (Newton Solver) to solve for w = 0
        # 'g' represents the system of equations to solve, 'x' is the variable
        nlp = {'x': z, 'g': w}
        self.newton_solver = cas.rootfinder('newton_solver', 'newton', nlp)

    def J(self, x, y):
        """
        Engineered objective function to guarantee:
        NE (saddle) at (0,0) and Spurious Min at (0.5,0.5)
        """
        decay = cas.exp(-0.01 * (x**2 + y**2))
        term1 = (0.3 * x**2 + y)**2
        term2 = (0.5 * y**2 + x)**2
        return -decay * (term1 + term2)

    def get_trajectory(self, x0, y0, method='gd', dt=0.01, steps=800):
        """ Numerically integrate the chosen vector field using RK4 """
        if method == 'gd':
            f = self.f_gd
        elif method == 'lss':
            f = self.f_lss
        elif method == 'dftr':
            f = self.f_dftr
        else:
            raise ValueError("Method must be 'gd' or 'lss'")

        traj = np.zeros((steps, 2))
        z = np.array([[x0], [y0]])

        for i in range(steps):
            traj[i, :] = z.flatten()
            # Runge-Kutta 4 integration for smooth/accurate continuous time
            k1 = np.array(f(z))
            k2 = np.array(f(z + 0.5 * dt * k1))
            k3 = np.array(f(z + 0.5 * dt * k2))
            k4 = np.array(f(z + dt * k3))
            z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return traj

    def plot(self, start_x, start_y):
        """ Generate proposal-quality plot of the cost landscape and trajectories """
        # Removed the seaborn style dependency. Using universal defaults.
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

        # 1. Plot the cost landscape
        grid_range = np.linspace(-15, 5, 150)
        X, Y = np.meshgrid(grid_range, grid_range)

        J_func = cas.Function('J', [self.x, self.y], [self.J_val])
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = J_func(X[i, j], Y[i, j]).full()[0, 0]

        # Gentle color grading with contour lines
        cp = ax.contourf(X, Y, Z, levels=60, cmap='RdBu_r', alpha=0.55)
        # ax.contour(X, Y, Z, levels=20, colors='black', linewidths=0.3, alpha=0.5)
        fig.colorbar(cp, ax=ax, label='Cost Function $J(x,y)$')

        # 2. Mark the local saddle point and local minimal
        ax.plot(-12.476, -8.677, 'k*', markersize=14, label='Nash Equilibrium')
        ax.plot(-1.317, -1.224, 'kX', markersize=11, label='Non-Nash Stationary Point')

        # 3. Simulate and plot trajectories

        steps = 2000
        dt = 1e-3
        traj_gd = self.get_trajectory(start_x, start_y, method='gd', dt=dt, steps=steps)
        traj_lss = self.get_trajectory(start_x, start_y, method='lss', dt=dt, steps=steps)
        traj_dftr = self.get_trajectory(start_x, start_y, method='dftr', dt=dt, steps=steps)

        # Using standard hex colors that look good on default backgrounds
        ax.plot(traj_gd[:, 0], traj_gd[:, 1], color='#d62728', linestyle='--',
                linewidth=2.5, label='Gradient Descent')
        ax.plot(traj_lss[:, 0], traj_lss[:, 1], color='#1f77b4', linestyle='-',
                linewidth=2.5, label='Local Symplectic Surgery')
        ax.plot(traj_dftr[:, 0], traj_dftr[:, 1], color="#0d8320", linestyle='-',
                linewidth=2.5, label='double Follow The Ridge')

        # Starting point marker
        ax.plot(start_x, start_y, 'ko', markersize=6)
        ax.annotate('Start', (start_x + 0.03, start_y), fontsize=10)

        # Formatting
        ax.set_xlim([grid_range[0], grid_range[-1]])
        ax.set_ylim([grid_range[0], grid_range[-1]])
        ax.set_xlabel('$x$ (Player 1)', fontsize=12)
        ax.set_ylabel('$y$ (Player 2)', fontsize=12)
        # ax.set_title('Solver Trajectory: Gradient Descent vs. LSS', fontsize=14, pad=10)
        ax.legend(loc='upper left', frameon=True, shadow=True)
        ax.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.show()
        return

        plt.plot(traj_gd[:, 0], linestyle='--')
        plt.plot(traj_gd[:, 1], linestyle='--')
        plt.plot(traj_lss[:, 0])
        plt.plot(traj_lss[:, 1])
        plt.show()

    def get_Dw(self, x_val, y_val):
        """ Calculate and return Dw matrix given numeric x and y """
        z_val = cas.vertcat(x_val, y_val)
        return self.eval_Dw(z_val).full()

    def solve_stationary_point(self, x_guess, y_guess):
        """ Solves w=0 using CasADi's Newton method from an initial guess """
        z_guess = cas.vertcat(x_guess, y_guess)

        # Run CasADi newton solver
        sol = self.newton_solver(x0=z_guess)
        z_star = sol['x'].full().flatten()
        x_star, y_star = z_star[0], z_star[1]

        # Extract Jxx and Jyy from Dw
        # Note: Dw = [[ Jxx,  Jxy],
        #             [-Jyx, -Jyy]]
        Dw_star = self.get_Dw(x_star, y_star)
        Jxx = Dw_star[0, 0]
        Jyy = -Dw_star[1, 1]

        return x_star, y_star, Jxx, Jyy

    def debug(self, start_x, start_y):
        z_val = cas.vertcat(start_x, start_y)
        grad = self.eval_w(z_val).full()
        print(f'grad at start: {grad}')
        print(self.solve_stationary_point(-12, -10))
        print(self.solve_stationary_point(-1.32, -1.2))


if __name__ == '__main__':
    model = LSS()
    start_x, start_y = -5, -10
    model.debug(start_x, start_y)
    model.plot(start_x, start_y)
