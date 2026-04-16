# Visualize car sprite size and collision function shape
import os
import numpy as np
from math import degrees, sin, cos
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import matplotlib.image as mpimg
from rd3g.utilities.util import BASEDIR, resolve_logname

car_img = mpimg.imread(os.path.join(BASEDIR, 'rd3g', 'resources', 'porsche_red.png'))
car_scale = 0.0045 / 2
car_pose = [0, 0, 0]  # x,y,heading
fig, ax = plt.subplots()

# Draw car sprite
rotated_car_img = np.clip(
    rotate(car_img,
           degrees(car_pose[2]), reshape=True), 0.0, 1.0)
L, W, _ = rotated_car_img.shape
im = ax.imshow(rotated_car_img,
               extent=[
                   car_pose[0] - W * car_scale,
                   car_pose[0] + W * car_scale,
                   car_pose[1] - L * car_scale,
                   car_pose[1] + L * car_scale
               ])

# Draw collision
# Calculate centers in world frame
cx = car_pose[0]
cy = car_pose[1]
theta = car_pose[2]
offset = 0.7
r = 0.7

# Circle 1 (Front)
c1_x = cx + offset * cos(theta)
c1_y = cy + offset * sin(theta)

# Circle 2 (Rear)
c2_x = cx - offset * cos(theta)
c2_y = cy - offset * sin(theta)

# Create Patches
# Alpha=0.5 allows you to see how well the circles cover the sprite
circle_front = patches.Circle((c1_x, c1_y), radius=r,
                              edgecolor='cyan', facecolor='cyan',
                              alpha=0.4, linewidth=2, label='Front Col')
circle_rear = patches.Circle((c2_x, c2_y), radius=r,
                             edgecolor='lime', facecolor='lime',
                             alpha=0.4, linewidth=2, label='Rear Col')

ax.add_patch(circle_front)
ax.add_patch(circle_rear)

# --- 3. Formatting ---
ax.set_aspect('equal')  # Critical for visual verification of collision shapes
ax.grid(True)
ax.set_xlim(-2.5, 2.5)  # Adjust zoom as needed
ax.set_ylim(-2.5, 2.5)

plt.show()
