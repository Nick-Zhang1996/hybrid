""" Generate CasADi cpp source code for all games """
import logging
from rd3g.games.car_merge_kinematic_bicycle_casadi import create_random_game as create_merge_game
from rd3g.games.intersection_casadi import create_random_game as create_intersection_game
from rd3g.games.car_racing_casadi import create_random_game as create_racing_game
from rd3g.utilities.casadi_util import generate_code
from rd3g.solvers.interior_point_game import InteriorPointGame, InteriorPointGameConfig

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _generate_both_variants(create_game, car_count, horizon, game_name):
    for variational_gne in [False, True]:
        logger.info(f'Codegen for {game_name}, {car_count=}, {horizon=}, {variational_gne=}...')
        game = create_game(car_count=car_count, horizon=horizon, variational_gne=variational_gne)
        solver_config = InteriorPointGameConfig(variational_gne=variational_gne)
        solver = InteriorPointGame(solver_config, game)
        generate_code(solver)
        logger.info('Done!')


def codegen_merge(car_count, horizon):
    _generate_both_variants(create_merge_game, car_count, horizon, 'merge')


def codegen_intersection(car_count, horizon):
    _generate_both_variants(create_intersection_game, car_count, horizon, 'intersection')


def codegen_racing(car_count, horizon):
    _generate_both_variants(create_racing_game, car_count, horizon, 'racing')


if __name__ == "__main__":
    # codegen_racing(8, 10)
    codegen_racing(4, 10)
    for T in [20]:
        for i in range(3, 4):
            codegen_merge(i, T)
            codegen_intersection(i, T)
