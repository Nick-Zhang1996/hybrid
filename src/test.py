import numpy as np
from time import time

import build.particle_game

A = np.eye(3)
B = np.ones((3,2))
x = np.array([1,2,3]).reshape((-1,1))
u = np.array([7,8]).reshape((-1,1))

def f(x,u):
    return A @ x + B @ u

game = build.particle_game.ParticleGame(1,1,1,1, 0.1,0.1,0.1,0.1,0.1, A,A,A,A,B,A)
t0 = time()

t0 = time()
result = game.f(x,u)
print(time()-t0)
print(result.shape)

t0 = time()
result = f(x,u)
print(result.shape)
print(time()-t0)

a = np.ones(10)
print(a.shape)
game.print_dim(a)
b = np.ones((10,11))
print(b.shape)
game.print_dim(b)
c = np.ones((10,11,12))
print(c.shape)
game.print_dim(c)



