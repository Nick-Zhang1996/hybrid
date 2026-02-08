""" Generate CasADi cpp source code for all games """
import logging
from rd3g.games.car_merge_kinematic_bicycle_casadi import create_random_game
from rd3g.utilities.casadi_util import generate_code
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def codegen(car_count, horizon):
    logger.info(f'Codegen for {car_count=}, {horizon=}...')
    game = create_random_game(car_count=car_count, horizon=horizon)
    solver_config = RD3GCasadiConfig()
    solver = RD3GCasadi(solver_config, game)
    generate_code(solver)
    logger.info('Done!')


if __name__ == "__main__":
    # codegen(5, 40)
    for T in [20]:
        for i in range(2, 9):
            codegen(i, T)
