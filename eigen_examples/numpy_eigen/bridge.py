# example program to call functions in c++/Eigen

import numpy as np
import example

a = np.linspace(0,100,100).reshape(10,10)
b = np.linspace(10,100,100).reshape(10,10)

def fun_np(a,b):
    return a @ b

def fun_cpp(a,b):
    return example.fun(a,b)

val = np.linalg.norm(fun_np(a,b))
print(val)
val = np.linalg.norm(fun_cpp(a,b))
print(val)


