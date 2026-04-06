# process image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = cv2.imread('../resources/ship.png')

# fill in the black contours
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = (gray < 254).astype('uint8')

plt.imshow(gray)
plt.show()

#result = cv2.bitwise_and(img, mask)
img = mpimg.imread('../resources/ship.png')
img[:,:,3] = np.array(gray,dtype=float)

plt.imshow(img)
plt.show()
plt.imsave('../resources/ship_alpha.png',img)
