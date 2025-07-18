# apply CarRacing in a receding horizon style
# NOTE for some reason, for the car racing problem the cpp version has trouble getting good solutions
from examples.car_racing import CarRacing
from math import radians, degrees
import matplotlib.pyplot as plt
import numpy as np
from residual_game import ResidualGameConfig


class CarRacingMpc(CarRacing):

    def __init__(self, config: ResidualGameConfig):
        super().__init__()
        self.T = 40

        self.guess = np.zeros((self.T, self.N, self.m))
        self.guess[:, 0, 0] = -radians(0)

    def simulate(self):
        overlap_steps = self.T // 2
        original_x0 = self.x0.copy()

        x_vec = [self.x0[np.newaxis, :, :]]
        u_vec = []
        # set  x0, u_ref

        for i in range(7):
            # find solution
            self.init()
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
        self._animation(u_vec, X=x_vec, gif_prefix='car_racing_mpc')
        self._visualize(u_vec, x=x_vec)
        plt.show()


if __name__ == "__main__":
    main = CarRacingMpc(ResidualGameConfig())
    main.setup()
    main.simulate()
