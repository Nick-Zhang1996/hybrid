import numpy as np

def jacobianNumerical(fun,x,dim=1):
    '''
    find jacobian of fun at x
    x: np.array  .shape = (n)
    fun: lambda function, f:R^n -> R^dim,
    return: Jacobian matrix of f'(x), shape = (1,n)
    '''
    epsilon = 1e-6
    x = np.array(x)
    n = x.shape[0]

    # A = df/dx
    A = np.zeros((dim,n),dtype=float)
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
            A[:,i] += (x_post_r - x_post_l) / (2*epsilon)
        else:
            A[:,i] += (x_post_r.flatten() - x_post_l.flatten()) / (2*epsilon)

    return A

# find jacobian and hessian for f(x): n->scalar
def hessianNumerical(fun,x):
    '''
    find hessian of fun at x
    x: np.array  .shape = (n)
    fun: lambda function, f:R^n -> R,
    return: Hessian matrix of f'(x), shape = (n,n)
    '''
    epsilon = 1e-5
    x = np.array(x)
    n = x.shape[0]
    H = np.zeros((n,n),dtype=float)
    return jacobianNumerical(lambda val:jacobianNumerical(fun,val),x,dim=n)

# find the directional derivative
def dirDer(fun,x0,dx):
    step = dx/np.linalg.norm(dx)*1e-6
    return (fun(x0+step)-fun(x0))/1e-6

def jacobianNumericalSlow(fun,x):
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
    A = np.zeros((1,n),dtype=float)
    I = np.eye(n)
    # find A
    for i in range(n):
        # d x / d x_i, ith row in A
        x_neg = fun(x - epsilon*I[i])
        x_pos = fun(x + epsilon*I[i])
        A[:,i] += (x_pos - x_neg) / (2*epsilon)

    return A

