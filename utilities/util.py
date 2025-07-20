import os
import inspect
import numpy as np

global BASEDIR
BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def printCurrentMemoryUsage(text=''):
    # Open the /proc/self/status file
    with open("/proc/self/status", "r") as file:
        # Read the file line by line
        for line in file:
            # Look for the line that starts with 'VmSize:'
            if line.startswith("VmSize:"):
                # Print the line (memory usage)
                print(text + ' ' + line.split()[1] + 'kB')
                break


def ifprint(*objects):
    if (PRINT):
        print(*objects)


def jacobianNumerical(fun, x, dim=1):
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
    if (dim == 0):
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

        if (dim == 1):
            A[:, i] += (x_post_r.item() - x_post_l.item()) / (2 * epsilon)
        else:
            A[:,
              i] += (x_post_r.flatten() - x_post_l.flatten()) / (2 * epsilon)

    return A


# find jacobian and hessian for f(x): n->scalar
def hessianNumerical(fun, x):
    '''
    find hessian of fun at x
    x: np.array  .shape = (n)
    fun: lambda function, f:R^n -> R,
    return: Hessian matrix of f'(x), shape = (n,n)
    '''
    epsilon = 1e-5
    x = np.array(x)
    n = x.shape[0]
    H = np.zeros((n, n), dtype=float)
    return jacobianNumerical(lambda val: jacobianNumerical(fun, val), x, dim=n)


# find the directional derivative
def dirDer(fun, x0, dx):
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


class PrintObject:
    silent = False
    debug = False

    def __init__(self):
        #print_ok(self.prefix() + "in use")
        #self.DEBUG = False
        pass

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
        if (self.silent):
            return
        # green
        print('\033[92m', self.prefix(), *message, '\033[0m')

    def print_debug(self, *message):
        if (self.silent):
            return
        # yellow
        if (self.config.DEBUG):
            print('\033[93m', self.prefix(),
                  inspect.stack()[1][3], *message, '\033[0m')

    def print_warning(self, *message):
        if (self.silent):
            return
        # yellow
        #print('\033[93m',self.prefix(), *message, '\033[0m')
        # red
        print('\033[91m', self.prefix(), 'WARNING: ', *message, '\033[0m')

    def print_info(self, *message):
        if (self.silent):
            return
        # light blue
        print('\033[96m', self.prefix(), *message, '\033[0m')
