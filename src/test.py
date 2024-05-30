import numpy as np

import build.particle_game

A = np.eye(3)
B = np.ones((3,2))
x = np.array([1,2,3]).reshape((-1,1))
u = np.array([7,8]).reshape((-1,1))

game = build.particle_game.ParticleGame()
game.set_A(A)
game.set_B(B)
result = game.f(x,u)
print(result)
print(A @ x + B @ u)
