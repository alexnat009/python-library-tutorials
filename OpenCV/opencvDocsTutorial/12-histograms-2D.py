import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("../images/toba.jpg")
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

# h, s, v = cv.split(img)
# hist, xbins, ybins = np.histogram2d(h.ravel(), s.ravel(), [180, 256], [[0, 180], [0, 256]])
fig, ax = plt.subplots(1, 2, figsize=(3, 3))
ax[0].imshow(img)
ax[1].imshow(hist, interpolation='nearest')
plt.show()
