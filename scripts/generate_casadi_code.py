""" Generate CasADi cpp source code for all games """
import logging
from rd3g.games.car_merge_kinematic_bicycle_casadi import create_random_game as create_merge_game
from rd3g.games.intersection_casadi import create_random_game as create_intersection_game
from rd3g.games.car_racing_casadi import create_random_game as create_racing_game
from rd3g.utilities.casadi_util import generate_code
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def codegen_merge(car_count, horizon):
    logger.info(f'Codegen for {car_count=}, {horizon=}...')
    game = create_merge_game(car_count=car_count, horizon=horizon)
    solver_config = RD3GCasadiConfig()
    solver = RD3GCasadi(solver_config, game)
    generate_code(solver)
    logger.info('Done!')


def codegen_intersection(car_count, horizon):
    logger.info(f'Codegen for {car_count=}, {horizon=}...')
    game = create_intersection_game(car_count=car_count, horizon=horizon)
    solver_config = RD3GCasadiConfig()
    solver = RD3GCasadi(solver_config, game)
    generate_code(solver)
    logger.info('Done!')


def codegen_racing(car_count, horizon):
    logger.info(f'Codegen for {car_count=}, {horizon=}...')
    game = create_racing_game(car_count=car_count, horizon=horizon)
    solver_config = RD3GCasadiConfig()
    solver = RD3GCasadi(solver_config, game)
    generate_code(solver)
    logger.info('Done!')


if __name__ == "__main__":
    codegen_racing(4, 20)
    codegen_merge(4, 40)
    codegen_intersection(4, 40)
    for T in [20]:
        for i in range(2, 9):
            # codegen_merge(i, T)
            # codegen_intersection(i, T)
            pass
