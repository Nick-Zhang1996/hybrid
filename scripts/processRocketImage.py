# process image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = cv2.imread('../resources/rocket.png')
'''
mask_R = np.logical_and(img[:,:,0] > 0.932, img[:,:,0] < 0.934)
mask_G = np.logical_and(img[:,:,1] > 0.932, img[:,:,1] < 0.934)
mask_B = np.logical_and(img[:,:,2] > 0.932, img[:,:,2] < 0.934)
mask = np.logical_and(mask_R, mask_G)
mask = np.logical_and(mask, mask_B)
mask = mask.astype('uint8')
plt.imshow(mask)
plt.show()

erosion_shape = cv2.MORPH_RECT
erosion_size = 1
element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
mask = cv2.erode(mask, element)
plt.imshow(mask)
plt.show()

dilation_shape = cv2.MORPH_RECT
dilation_size = 1
element = cv2.getStructuringElement(dilation_shape, (2 * dilation_size + 1, 2 * dilation_size + 1), (dilation_size, dilation_size))
mask = cv2.dilate(mask, element)
'''

# fill in the black contours
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = (gray < 5).astype('uint8')

plt.imshow(gray)
plt.show()

contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros_like(gray)
for contour in contours:
    mask = cv2.drawContours(mask, [contour], -1, (255,255,255), -1)
plt.imshow(mask)
plt.show()

#result = cv2.bitwise_and(img, mask)
img = mpimg.imread('../resources/rocket.png')
img[:,:,3] = np.array(mask,dtype=float)/255.0

plt.imshow(img)
plt.show()
breakpoint()
plt.imsave('./resources/rocket_alpha.png',img)
