import numpy as np
from examples.unstructured_driving import UnstructuredDriving
from residual_game import ResidualGameConfig


class CarMergeDoubleIntegrator(UnstructuredDriving):

    def __init__(self, config: ResidualGameConfig, car_count: int = 8):
        #super().__init__(car_count=5)
        # simplest, 3 car
        #super().__init__(car_count=3)
        #self.x0 = np.array([[3.0,0, 2.0, 0], [0.0, 0, 2.0, 0], [1.5, 1.5, 2.0, 0]])
        #self.target_y = [0,0, 0]

        # merging that require rear car to slow down, 3 car
        #super().__init__(car_count=3)
        #self.x0 = np.array([[3.0,0, 2.1, 0], [1.0, 0, 2.1, 0], [1.5, 1.5, 2.0, 0]])
        #self.target_y = [0,0, 0]

        # collision resolution, longitudinal, 2 car Dr 6ms
        #super().__init__(car_count=2)
        #self.x0 = np.array([[1.0, 0, 2.3, 0], [1.5, 0, 2.2, 0]])
        #self.target_y = [0, 0]

        # complicated, zipper merge, car_count: main_lane_n + merge_lane_n, Dr 650ms
        np.random.seed(0)
        main_lane_n = min(int(0.65 * car_count), car_count - 1)
        merge_lane_n = car_count - main_lane_n
        super().__init__(car_count=main_lane_n + merge_lane_n)

        x_pos_main_lane = np.linspace(
            0, (main_lane_n - 1) * 2.5,
            main_lane_n) + np.random.random(main_lane_n)
        x_pos_merge_lane = 1.0 + np.linspace(
            0, (merge_lane_n - 1) * 2.5,
            merge_lane_n) + np.random.random(merge_lane_n)
        v_main_lane = 2.0 + np.random.random(main_lane_n)
        v_merge_lane = 2.0 + np.random.random(merge_lane_n)
        x0_main_lane = np.vstack([
            x_pos_main_lane,
            np.zeros(main_lane_n), v_main_lane,
            np.zeros(main_lane_n)
        ]).T
        x0_merge_lane = np.vstack([
            x_pos_merge_lane, 1.5 * np.ones(merge_lane_n), v_merge_lane,
            np.zeros(merge_lane_n)
        ]).T

        self.T = 15
        self.guess = np.zeros((self.T, self.N, self.m))
        self.x0 = np.vstack([x0_main_lane, x0_merge_lane])
        self.target_y = [0] * (main_lane_n + merge_lane_n)
        self.J_Qr = np.diag([0, 1, 0.1, 0])
        self.J_Q = np.diag([0, 0, 0, 0.5])
        self.print_debug_enable()
        self.config.iterations = 30
        self.final_resolution = 1e-10


if __name__ == "__main__":
    main = CarMergeDoubleIntegrator(ResidualGameConfig(), 3)
    main.setup()
    main.solve()
    main.final()
