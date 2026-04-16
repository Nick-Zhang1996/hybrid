# apply CarDrift in a receding horizon style
from examples.one_car_drift import OneCarDrift
from math import radians, degrees
import matplotlib.pyplot as plt
import numpy as np
from residual_game import ResidualGameConfig


class OneCarDriftMpc(OneCarDrift):

    def __init__(self, config: ResidualGameConfig):
        super().__init__(config)
        self.T = 100
        self.Tmax = 0.174 * 0.4
        self.dt = dt = 0.05
        # initial state,
        #self.x0 = np.array([[0,0,radians(10),1,0.2,0.1, radians(10),10]])
        vx = 0.5
        vy = -0.38
        r = 0.1333
        theta = -radians(10.45)
        Br = radians(3.82)
        self.x0 = np.array([[0, 0, radians(17), vx, vy, r, theta, Br]])
        self.guess = np.zeros((self.T, self.N, self.m))
        self.guess[:, 0, 0] = -radians(0)

        self.mu_ref = radians(17)
        self.vx_ref = 1.0
        self.control_cost = 1e-2
        self.n_cost = 0.3
        self.vx_cost = 0.1

    def simulate(self):
        overlap_steps = self.T // 2
        original_x0 = self.x0.copy()

        x_vec = [self.x0[np.newaxis, :, :]]
        u_vec = []
        # set  x0, u_ref

        for i in range(20):
            # find solution
            u_ref, full_x_ref, has_converged = self.solve(save_gif=False,
                                                          visualize=False,
                                                          animate=False)
            # log state/control, move horizon forward
            x_vec.append(full_x_ref[1:overlap_steps + 1])
            u_vec.append(u_ref[:overlap_steps])
            self.init()  # reset solver dynamic parameters
            self.x0 = full_x_ref[overlap_steps]
            self.guess = np.zeros((self.T, self.N, self.m))
            self.guess[:overlap_steps] = u_ref[overlap_steps:]

        self.x0 = original_x0
        x_vec = np.vstack(x_vec)
        u_vec = np.vstack(u_vec)
        self.T = len(u_vec)
        vi = (x_vec[0, 0, 3]**2 + x_vec[0, 0, 4]**2)**0.5
        vf = (x_vec[-1, 0, 3]**2 + x_vec[-1, 0, 4]**2)**0.5
        print(f' vi {vi:.2f}, vf {vf:.2f}')

        self._visualize(u_vec, x=x_vec)
        plt.show()

        #self._visualize(u_vec)
        #plt.show()
        v_total = (x_vec[:, 0, 3]**2 + x_vec[:, 0, 4]**2)**0.5
        print(v_total)
        #breakpoint()


if __name__ == "__main__":
    main = OneCarDriftMpc(ResidualGameConfig())
    main.setup()
    main.simulate()
    '''
    u_vec = np.zeros((main.T*10, main.N, main.m))
    main.T = len(u_vec)
    main._visualize(u_vec)
    plt.show()

    X = np.vstack([main.x0[np.newaxis,:,:],main.rollout(main.x0,u_vec)])
    name = ['s','n','mu','vx','vy','r']
    for i in range(len(name)):
        fig,ax = plt.subplots()
        ax.plot(X[:,0,i])
        ax.set_title(name[i])
        plt.show()
    breakpoint()
    '''
