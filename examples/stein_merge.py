""" demo to show SVG can find two equilibriums"""
import logging
from time import time
import numpy as np

from residual_game import ResidualGame, ResidualGameConfig
from stein_game import SteinGame, SteinGameConfig
from examples.car_merge_kinematic_bicycle import CarMergeKinematicBicycle
from src.build.car_merge_kinematic_bicycle import CarMergeKinematicBicycle as cpp_CarMergeKinematicBicycle  # pylint: disable=no-name-in-module

logger = logging.getLogger("SteinMerge")
logger.setLevel(logging.INFO)


class SteinMerge(CarMergeKinematicBicycle):
    """ Wrapper for SteinGame version of CarMergeKinematicBicycle"""

    def __init__(self, config: SteinGameConfig, car_count: int):
        """
        super().__init__(car_count=2)
        main_lane_n = 1
        merge_lane_n = 2 - main_lane_n
        x_pos_main_lane = np.array([1.5])
        x_pos_merge_lane = np.array([1.5])
        v_main_lane = 2.0
        v_merge_lane = 2.0
        x0_main_lane = np.vstack([
            x_pos_main_lane, self.track_width / 2 * np.ones(main_lane_n) + 0.2,
            v_main_lane,
            np.zeros(main_lane_n)
        ]).T
        x0_merge_lane = np.vstack([
            x_pos_merge_lane,
            -self.track_width / 2 * np.ones(merge_lane_n) - 0.2, v_merge_lane,
            np.zeros(merge_lane_n)
        ]).T
        self.x0 = np.vstack([x0_main_lane, x0_merge_lane])
        self.target_y = [1] * (main_lane_n + merge_lane_n)
        """
        super().__init__(config, car_count, 20)
        assert isinstance(self, SteinGame)
        self.guess = np.zeros((self.T, self.N, self.m))
        # multiple car merge, car_count: main_lane_n + merge_lane_n
        main_lane_n = min(int(0.67 * car_count), car_count - 1)
        merge_lane_n = car_count - main_lane_n
        x_pos_main_lane = np.linspace(
            0, (main_lane_n - 1) * 5.4,
            main_lane_n) + np.random.random(main_lane_n)
        x_pos_merge_lane = np.linspace(
            0, (merge_lane_n - 1) * 5.4,
            merge_lane_n) + np.random.random(merge_lane_n)
        v_main_lane = 2.0 + np.random.random(main_lane_n) * 0
        v_merge_lane = 2.0 + np.random.random(merge_lane_n) * 0
        x0_main_lane = np.vstack([
            x_pos_main_lane, self.track_width / 2 * np.ones(main_lane_n),
            v_main_lane,
            np.zeros(main_lane_n)
        ]).T
        x0_merge_lane = np.vstack([
            x_pos_merge_lane, -self.track_width / 2 * np.ones(merge_lane_n),
            v_merge_lane,
            np.zeros(merge_lane_n)
        ]).T
        self.x0 = np.vstack([x0_main_lane, x0_merge_lane])
        self.target_y = [1] * (main_lane_n + merge_lane_n)
        self.config.stein_iterations = 20
        self.config.particles = 20
        self.cpp = None

    def setup(self):
        """ Instantiate cpp/eigen module and set initial condition"""
        if (self.config.USE_CPP or self.config.CPP_DEBUG):
            self.cpp = cpp_CarMergeKinematicBicycle(
                self.N, self.T, self.dt, self.rho, self.rho_b, self.bc_a,
                self.bc_b, self.config.tolerance, self.backtracking_max_iter,
                self.J_Qr, self.J_Q, self.J_R, self.h_Qh, self.target_y,
                self.collision_radius, self.config.iterations, False)
            self.cpp.set_x0(self.x0)

    def TestSolve(self, save_gif=False, visualize=False, animate=False):
        dim_u = (self.T, self.N, self.m)
        _u_ref = np.zeros(dim_u)
        # u_ref[:,1,0] = 0.5
        # u_ref[:,0,0] = -0.4
        # u_ref[:,1,0] = -0.5
        # u_ref[:,0,0] = 0.4

        _u_ref, _full_x_ref, _has_converged = ResidualGame.solve(
            self,
            u_ref=_u_ref.reshape(dim_u),
            save_gif=save_gif,
            visualize=visualize,
            animate=animate,
        )
        return _u_ref, _full_x_ref, _has_converged


if __name__ == '__main__':
    # Solve game for random initial states
    # np.random.seed(2)
    car_count = 5
    main = SteinMerge(SteinGameConfig(), car_count=car_count)
    main.setup()
    t0 = time()
    u_ref, full_x_ref, has_converged = main.solve()
    dt = time() - t0
    logger.info(f'Solving {car_count} car merging with {dt=} seconds')
    main.final()
    logger.info(
        f'u_ref mean {np.mean(u_ref.flatten())} std {np.std(u_ref.flatten())}')
    # main.testAnimation()

    # Visualize top 5 weighted equilibria
    #for i in range(5):
    #    main.visualize(U=main.belief_support[i],
    #                   visualize=True,
    #                   animate=True,
    #                   gif_prefix=f'car_{car_count}_case_{i}')

    # Test Stein game's prediction
    # The Nash Equilibrium chosen by each agent
    # NOTE: we need to limit the index range to avoid choosing high-residual samples
    chosen_id_by_agent = np.random.randint(0,
                                           len(main.belief_support),
                                           size=car_count)
    logger.info(chosen_id_by_agent)
    realized_u_ref_by_agent = []
    for i, ne_id in enumerate(chosen_id_by_agent):
        realized_u_ref_by_agent.append(main.belief_support[ne_id].reshape(
            (main.T, main.N, main.m))[:, i, :])
    realized_u_ref = np.stack(realized_u_ref_by_agent, axis=1)
    assert realized_u_ref.shape == (main.T, main.N, main.m)
    noise = np.random.normal(loc=0, scale=1e-1, size=realized_u_ref.shape)
    noisy_u_ref = realized_u_ref + noise

    # Visualize the uncoordinated trajectory
    main.visualize(u=realized_u_ref, gif_prefix='uncoordinated')

    # TODO: find collision count for uncoordinated case
    # Simulate game, update
    for k in range(main.T // 2):
        main.updateBelief(u_k=noisy_u_ref[k, :, :], k=k)

        # Find statistics on how many agent's intent is correctly estimated
        correct_estimate_count: int = 0
        for agent in range(car_count):
            high_likelihood_indices = np.argsort(
                main.belief_weight_by_agent[agent])[-3:]
            if chosen_id_by_agent[agent] in high_likelihood_indices:
                correct_estimate_count += 1
        logger.info(
            f'step {k}, correct estimate {correct_estimate_count}/{car_count}')
