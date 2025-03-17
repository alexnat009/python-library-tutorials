import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../images/smarties.jpg", 0)

_, th1 = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
kernel = np.ones((2, 2), np.uint8)
k = 1
dilation = cv2.dilate(th1, kernel, iterations=k)
erosion = cv2.erode(th1, kernel, iterations=k)
opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(th1, cv2.MORPH_GRADIENT, kernel)
titles = ["image", "mask", "dilation", "erosion", "opening", "closing", "gradient"]
images = [img, th1, dilation, erosion, opening, closing, gradient]

for i in range(7):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
