import numpy as np
from time import time
from examples.unstructured_driving import UnstructuredDriving
from residual_game import ResidualGameConfig
import matplotlib.pyplot as plt


class CarCrossingDoubleIntegrator(UnstructuredDriving):

    def __init__(self, config: ResidualGameConfig, car_count=4):
        super().__init__(config, car_count=car_count)
        self.T = 15
        v_car_count = int(car_count / 2)
        h_car_count = car_count - v_car_count
        # x,y,vx,vy
        self.x0 = np.vstack([[-3.0, 0, 2.0, 0], [-1.0, 0, 2.0, 0],
                             [0, -2.0, 0, 2], [0, -5.0, 0, 2]])
        # vx, vy
        self.target_v = np.vstack([[2.0, 0] * v_car_count +
                                   [0, 2.0] * h_car_count])

        x_pos_v_car = np.linspace(-(v_car_count - 1) * 2.5, -2,
                                  v_car_count) + np.random.random(v_car_count)
        y_pos_h_car = np.linspace(-(h_car_count - 1) * 2.5, -2,
                                  h_car_count) + np.random.random(h_car_count)
        vel_v_car = 2.0 + np.random.random(v_car_count) / 2
        vel_h_car = 2.0 + np.random.random(h_car_count) / 2
        x0_v_car = np.vstack([
            x_pos_v_car,
            np.zeros(v_car_count), vel_v_car,
            np.zeros(v_car_count)
        ]).T
        x0_h_car = np.vstack([
            np.zeros(h_car_count), y_pos_h_car,
            np.zeros(h_car_count), vel_h_car
        ]).T
        self.x0 = np.vstack([x0_v_car, x0_h_car])

        # v_cars go in x direction, y is target lane
        self.v_cars = list(range(car_count))[:v_car_count]
        self.target_y = [0] * car_count
        self.h_cars = list(range(car_count))[v_car_count:]
        self.target_x = [0] * car_count

        self.visual_x_lim = [-10, 10]
        self.visual_y_lim = [-10, 10]

        self.J_x_ref_fun = lambda i: np.array([
            0, self.target_y[i], 2.0, 0
        ]) if i in self.v_cars else np.array([self.target_x[i], 0, 0, 2.0])
        self.J_Qr_fun = lambda i: np.diag(
            [0, 1, 1, 0]) if i in self.v_cars else np.diag([1, 0, 0, 1])
        self.J_Q_fun = lambda i: np.diag(
            [0, 0, 0, 1e-2]) if i in self.v_cars else np.diag([0, 0, 1e-2, 0])
        self.J_R = np.eye(self.m) * 1e-2
        #(x-self.J_x_ref_fun(i)).T @ self.J_Qr @ (x-self.J_x_ref_fun(i)) + x.T @ self.J_Q @ x + u.T @ self.J_R @ u
        self.guess = np.zeros((self.T, self.N, self.m))

    def _visualize(self, u, x=None):
        if (x is None):
            x = np.vstack(
                [self.x0[np.newaxis, :, :],
                 self.rollout(self.x0, u)])
        fig, ax = plt.subplots()
        #ax.vlines(x=-self.track_width/2,ymin=-1,ymax=self.track_length)
        #ax.vlines(x=self.track_width/2,ymin=-1,ymax=self.track_length)
        for i in range(self.N):
            xx = x[:, i, 0]
            yy = x[:, i, 1]
            plt.plot(yy, xx, '*-')
        ax.set_aspect('equal', adjustable='box')
        return fig

    def J(self, x, u, i):
        ''' 
        step cost for an agent, given x,u 
        x.shape (n) x = [x,y,vx,vy]
        u.shape (m) u = [ax, ay]
        i: agent id
        '''
        #return (x[2] - 2.0)**2 + (x[1] - self.target_y[i])**2 + 1e-2*x[3]**2 + 1e-2*u.T @ np.eye(self.m) @ u
        return (x - self.J_x_ref_fun(i)).T @ self.J_Qr_fun(i) @ (
            x - self.J_x_ref_fun(i)
        ) + x.T @ self.J_Q_fun(i) @ x + u.T @ self.J_R @ u

    def dJ_dx(self, x, u, i):
        return 2 * (x - self.J_x_ref_fun(i)
                    ).T @ self.J_Qr_fun(i) + 2 * x.T @ self.J_Q_fun(i)

    def dJ_du(self, x, u, i):
        return 2 * u.T @ self.J_R

    def dJ_dxdx(self, x, u, i):
        return 2 * self.J_Qr_fun(i) + 2 * self.J_Q_fun(i)


if __name__ == "__main__":
    main = CarCrossingDoubleIntegrator(ResidualGameConfig(), 10)
    t0 = time()
    main.solve()
    print(f'solution time: {time()-t0}')
    print(f'collisions: {main.violations}')
    main.final()
