''' Common utilities for entire project'''
# pylint: disable=invalid-name
import os
import inspect
import functools

import numpy as np
import pyttsx3

# root folder of repo.
BASEDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASEDIR, 'outputs')
BENCHMARK_PLOTS_DIR = os.path.join(OUTPUT_DIR, 'benchmarks')
PICS_DIR = os.path.join(OUTPUT_DIR, 'pics')
GIFS_DIR = os.path.join(OUTPUT_DIR, 'gifs')


engine = pyttsx3.init()


def talk(text):
    engine.setProperty('rate', 150)  # Speed in words per minute
    engine.say(text)
    engine.runAndWait()


def cpp_capable(py_function):
    """
    A decorator that toggles between a Python and C++ implementation.

    It assumes the class instance (`self`) has:
    1. A `self.config.USE_CPP` boolean attribute.
    2. A `self.cpp` object that contains the C++ functions with
       names and arguments identical to the Python methods.
    """
    @functools.wraps(py_function)
    def wrapper(self, *args, **kwargs):
        if self.USE_CPP:
            # Get the function with the same name from the C++ object
            cpp_function = getattr(self.cpp, py_function.__name__)
            return cpp_function(*args, **kwargs)
        else:
            return py_function(self, *args, **kwargs)
    return wrapper


def print_current_memory_usage(text=''):
    ''' Use the /proc/self/status file '''
    with open("/proc/self/status", "r", encoding='utf-8') as file:
        # Read the file line by line
        for line in file:
            # Look for the line that starts with 'VmSize:'
            if line.startswith("VmSize:"):
                # Print the line (memory usage)
                print(text + ' ' + line.split()[1] + 'kB')
                break


def resolve_logname(prefix='run', suffix='log'):
    """ Find next available logname, e.g. [prefix]_3.[suffix]"""
    no = 1
    if suffix == 'gif':
        log_folder = GIFS_DIR
    elif suffix in {'png', 'jpg', 'jpeg', 'pdf', 'svg'}:
        log_folder = PICS_DIR
    else:
        log_folder = os.path.join(BASEDIR, 'logs')
    os.makedirs(log_folder, exist_ok=True)
    filename = os.path.join(log_folder, f'{prefix}_{no}.{suffix}')
    while os.path.isfile(filename):
        no += 1
        filename = os.path.join(log_folder, f'{prefix}_{no}.{suffix}')

    return filename


def jacobian_numerical(fun, x, dim=1):
    """
    find jacobian of fun at x
    Args:
        x: np.array  .shape = (n)
        fun: lambda function, f:R^n -> R^dim,
        dim: dimension of output for fun
    Return:
        Jacobian matrix of f'(x), shape = (1,n)
    """
    epsilon = 1e-6
    x = np.array(x)
    n = x.shape[0]

    # A = df/dx
    A = np.zeros((dim, n), dtype=float)
    # empty function
    if dim == 0:
        return A
    # find A
    for i in range(n):
        # d x / d x_i, ith row in A
        x_l = x.copy()
        x_l[i] -= epsilon

        x_post_l = fun(x_l)

        x_r = x.copy()
        x_r[i] += epsilon
        x_post_r = fun(x_r)

        if dim == 1:
            A[:, i] += (x_post_r.item() - x_post_l.item()) / (2 * epsilon)
        else:
            A[:,
              i] += (x_post_r.flatten() - x_post_l.flatten()) / (2 * epsilon)

    return A


def hessian_numerical(fun, x):
    '''
    find hessian of fun at x
    x: np.array  .shape = (n)
    fun: lambda function, f:R^n -> R,
    return: Hessian matrix of f'(x), shape = (n,n)
    '''
    x = np.array(x)
    n = x.shape[0]
    return jacobian_numerical(lambda val: jacobian_numerical(fun, val), x, dim=n)


def dirDer(fun, x0, dx):
    ''' find the directional derivative '''
    step = dx / np.linalg.norm(dx) * 1e-6
    return (fun(x0 + step) - fun(x0)) / 1e-6


def jacobianNumericalSlow(fun, x):
    '''
    find jacobian of fun at x
    x: np.array  .shape = (n)
    fun: lambda function, f:R^n -> R,
    return: Jacobian matrix of f'(x), shape = (1,n)
    '''
    epsilon = 1e-6
    x = np.array(x)
    n = x.shape[0]

    # A = df/dx
    A = np.zeros((1, n), dtype=float)
    I = np.eye(n)
    # find A
    for i in range(n):
        # d x / d x_i, ith row in A
        x_neg = fun(x - epsilon * I[i])
        x_pos = fun(x + epsilon * I[i])
        A[:, i] += (x_pos - x_neg) / (2 * epsilon)

    return A


# pylint: disable=missing-function-docstring
class PrintObject:
    ''' Base class for printing, obselete'''
    silent = False
    debug = False

    def __init__(self):
        self.DEBUG = False

    def print_debug_enable(self):
        self.DEBUG = True

    def print_debug_disable(self):
        self.DEBUG = False

    def silent_mode_enable(self):
        self.silent = True

    def silent_mode_disable(self):
        self.silent = False

    def prefix(self):
        return "[" + self.__class__.__name__ + "]: "

    def print_error(self, *message):
        print('\033[91m', self.prefix(), 'ERROR ', *message, '\033[0m')
        raise RuntimeError

    def print_ok(self, *message):
        if self.silent:
            return
        # green
        print('\033[92m', self.prefix(), *message, '\033[0m')

    def print_debug(self, *message):
        if self.silent:
            return
        # yellow
        print('\033[93m', self.prefix(),
              inspect.stack()[1][3], *message, '\033[0m')

    def print_warning(self, *message):
        if self.silent:
            return
        # yellow
        # print('\033[93m',self.prefix(), *message, '\033[0m')
        # red
        print('\033[91m', self.prefix(), 'WARNING: ', *message, '\033[0m')

    def print_info(self, *message):
        if self.silent:
            return
        # light blue
        print('\033[96m', self.prefix(), *message, '\033[0m')
