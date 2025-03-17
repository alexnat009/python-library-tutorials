import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("../images/toba.jpg")
img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

hist = cv.calcHist(images=[img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
print(hist)
# plt.hist(hist, bins=25, range=[0, 256])
# plt.hist(img.ravel(), 256, [0, 256])
# plt.show()
colors = ('b', 'g', 'r')
for i, color in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
plt.show()