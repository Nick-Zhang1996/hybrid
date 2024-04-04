import numpy as np
# differentiate dynamics around nominal state and control
# return:f'(x) , where f is a R^n -> R
def linearizeNumerical(fun,x):
    epsilon = 1e-2
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

        A[:,i] += (x_post_r.flatten() - x_post_l.flatten()) / (2*epsilon)

    return A
