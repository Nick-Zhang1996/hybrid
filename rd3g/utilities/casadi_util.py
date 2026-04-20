""" Generate Casadi C code for a solver and game pair 

<module_name, N (agent number), and T (horizon)> gives a unique pair of .cpp, .h files 
These need to be compiled to a unique dll (.so) file, which is loaded during solver.init_cpp_backend()
"""
import logging
from casadi import *
from rd3g.utilities.util import BASEDIR
from scipy.sparse import csc_matrix, csc_array

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_code(solver):
    """ Generate CasADi C code for a solver and game with its specific config.
    CasADi expects fixed dimension, so the exact game config needs to be given.
    Each tuple of (GameType, N,T) requres a different source file.
    Changing solver configuration MAY require re-compilation. (TBD)
    TODO should part of this logic be in the solver?
    Args:
        solver: Solver instance, with solver.game set, and construct_casadi_fun() called
    """
    game = solver.game
    config = game.config

    # Switch working directory
    codegen_dir = os.path.join(BASEDIR, 'casadi_codegen')
    if not os.path.exists(codegen_dir):
        os.makedirs(codegen_dir)
    old_cwd = os.getcwd()
    os.chdir(codegen_dir)

    # Generate source code
    module_name = game.__module__.split('.')[-1]
    cg = CodeGenerator(f'{module_name}_N{config.N}_T{config.T}.cpp',
                       {'with_header': True})
    cg.add(solver.r_casadi)
    cg.add(solver.dr_dy_casadi)
    cg.add(solver.h_casadi)
    cg.add(solver.collision_h_casadi)
    cg.add(solver.rollout_casadi)
    cg.add(solver.get_n_fun)
    cg.add(solver.get_m_fun)
    cg.add(solver.get_context_casadi)
    cg.add(solver.get_full_context_casadi)
    for i in range(solver.N):
        cg.add(solver.Ki_casadi_vec[i])
    filename = cg.generate()
    logger.info(f'CasADi file generated at {filename}')

    os.chdir(old_cwd)
    return


def dm_to_csc(dm, array=False):
    """ Convert a CasADi DM sparse matrix to scipy csc_matris
    Args:
        dm: DM object,
        array: if True, return csc array, otherwise return csc matrix
    """

    data = np.array(dm.nonzeros(), dtype=np.float64).ravel()

    indices = np.array(dm.sparsity().row(), dtype=np.int32)
    indptr = np.array(dm.sparsity().colind(), dtype=np.int32)

    shape = dm.size()

    if array:
        return csc_array((data, indices, indptr), shape=shape, copy=False)
    else:
        return csc_matrix((data, indices, indptr), shape=shape, copy=False)
