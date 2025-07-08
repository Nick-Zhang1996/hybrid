""" setup car racing games benchmark"""

from math import radians
import numpy as np
import matplotlib.pyplot as plt

from examples.car_racing import CarRacing


# run simulation with CarRacing, in receding horizon fashion
# report winner
class CarRacingInstance(CarRacing):

    def __init__(self):
        super().__init__()
        self.T = 20
        self.guess = np.zeros((self.T, self.N, self.m))
        self.print_debug_disable()

    def simulate(self):
        overlap_steps = self.T // 2
        original_x0 = self.x0.copy()

        x_vec = [self.x0[np.newaxis, :, :]]
        u_vec = []
        # set  x0, u_ref
        rval = None

        for i in range(40):
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

            if (np.max(x_vec[-1][-1, :, 0]) > self.track.raceline_len_m):
                print(f'result: {x_vec[-1][-1,:,0]}')
                rval = np.argmax(x_vec[-1][-1, :, 0])
                break
        x_vec = np.vstack(x_vec)
        u_vec = np.vstack(u_vec)
        print(x_vec[-1])
        return rval


if __name__ == "__main__":
    count = 100
    # car 0 is GT
    '''
    print(f'GT: front, slow')
    results = []
    for i in range(count):
        main = CarRacingInstance()
        noise = (2*np.random.rand(2,4) - 1 )* np.array([0.2,0.2,0.4,radians(10)])
        #main.x0 = np.array([[0.2, 1.0, 0.15, 0], [0, 1.1, -0.2, radians(0)]])
        main.x0 = np.array([[0.3, 1.0, 0, 0], [0, 1.2, 0, 0]]) + noise
        main.J_x_ref_fun = lambda i:np.array([0,1.0+i*0.2,0,0])
        main.setup()
        rval = main.simulate()
        results.append(rval)
    print(results)
    print(f'GT: front, slow: {1-np.sum(results)/20}')

    print(f'GT: rear, fast')
    results = []
    for i in range(count):
        main = CarRacingInstance()
        noise = (2*np.random.rand(2,4) - 1 )* np.array([0.2,0.2,0.4,radians(10)])
        #main.x0 = np.array([[0, 1.1, -0.2, radians(0)],[0.2, 1.0, 0.15, 0]])
        main.x0 = np.array([[0, 1.2, 0, 0], [0.3, 1.0, 0, 0]]) + noise
        main.J_x_ref_fun = lambda i:np.array([0,1.2-i*0.2,0,0])
        main.setup()
        rval = main.simulate()
        results.append(rval)
    print(results)
    print(f'GT: rear, fast: {1-np.sum(results)/20}')

    print(f'GT: front, fast')
    results = []
    for i in range(count):
        main = CarRacingInstance()
        noise = (2*np.random.rand(2,4) - 1 )* np.array([0.2,0.2,0.4,radians(10)])
        #main.x0 = np.array([[0, 1.1, -0.2, radians(0)],[0.2, 1.0, 0.15, 0]])
        main.x0 = np.array([[0.3, 1.2, 0, 0], [0.0, 1.0, 0, 0]]) + noise
        main.J_x_ref_fun = lambda i:np.array([0,1.2-i*0.2,0,0])
        main.setup()
        rval = main.simulate()
        results.append(rval)
    print(results)
    print(f'GT: front, fast: {1-np.sum(results)/20}')

    '''
    print(f'GT: rear, slow')
    results = []
    for i in range(count):
        main = CarRacingInstance()
        noise = (2 * np.random.rand(2, 4) - 1) * np.array(
            [0.2, 0.2, 0.4, radians(10)])
        #main.x0 = np.array([[0, 1.1, -0.2, radians(0)],[0.2, 1.0, 0.15, 0]])
        main.x0 = np.array([[0, 1.0, 0, 0], [0.3, 1.2, 0, 0]]) + noise
        main.J_x_ref_fun = lambda i: np.array([0, 1.0 + i * 0.2, 0, 0])
        main.setup()
        rval = main.simulate()
        results.append(rval)
    print(results)
    print(f'GT: rear, slow: {1-np.sum(results)/20}')
