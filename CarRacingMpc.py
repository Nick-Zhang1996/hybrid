# apply CarRacing in a receding horizon style
from CarRacing import CarRacing
from math import radians, degrees
import matplotlib.pyplot as plt
import numpy as np

class CarRacingMpc(CarRacing):
    def __init__(self):
        super().__init__()
        self.T = 30

        self.guess = np.zeros((self.T,self.N,self.m))
        self.guess[:,0,0] = -radians(0)

    def simulate(self):
        overlap_steps = self.T//2
        original_x0 = self.x0.copy()

        x_vec = [self.x0[np.newaxis,:,:]]
        u_vec = []
        # set  x0, u_ref

        for i in range(3): # 20 will to full circle
            # find solution
            u_ref, full_x_ref, has_converged = self.solve(save_gif=False, visualize=False, animate=False)
            # log state/control, move horizon forward
            x_vec.append(full_x_ref[1:overlap_steps+1])
            u_vec.append(u_ref[:overlap_steps])
            self.init() # reset solver dynamic parameters
            self.x0 = full_x_ref[overlap_steps]
            self.guess = np.zeros((self.T,self.N,self.m))
            self.guess[:overlap_steps] = u_ref[overlap_steps:]

        self.x0 = original_x0
        x_vec = np.vstack(x_vec)
        u_vec = np.vstack(u_vec)

        self._animation(u_vec, X=x_vec, gif_prefix='two_car_drift_mpc')
        self.T = len(u_vec)
        vi = (x_vec[0,0,3]**2 + x_vec[0,0,4]**2)**0.5
        vf = (x_vec[-1,0,3]**2 + x_vec[-1,0,4]**2)**0.5
        print(f' vi {vi:.2f}, vf {vf:.2f}')

        self._visualize(u_vec,X=x_vec)
        plt.show()

        #self._visualize(u_vec)
        #plt.show()
        v_total = (x_vec[:,0,3]**2 + x_vec[:,0,4]**2)**0.5
        print(v_total)
        breakpoint()


if __name__=="__main__":
    main = CarRacingMpc()
    main.setup()
    main.simulate()
