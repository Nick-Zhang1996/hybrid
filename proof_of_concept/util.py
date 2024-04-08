import numpy as np
# x: np.array.shape(n)
# given fun: f(x): n -> scalar
# return:f'(x) , where f is a R^n -> R
def linearizeNumerical(fun,x):
    epsilon = 5e-2
    x = np.array(x)
    n = x.shape[0]

    # A = df/dx
    A = np.zeros((1,n),dtype=np.float)
    # find A
    for i in range(n):
        # d x / d x_i, ith row in A
        x_l = x.copy()
        x_l[i] -= epsilon

        x_post_l = fun(x_l)

        x_r = x.copy()
        x_r[i] += epsilon
        x_post_r = fun(x_r)

        A[:,i] += (x_post_r - x_post_l) / (2*epsilon)

    return A

# find the directional derivative
def dirDer(fun,x0,dx):
    step = dx/np.linalg.norm(dx)*1e-6
    return (fun(x0+step)-fun(x0))/1e-6

