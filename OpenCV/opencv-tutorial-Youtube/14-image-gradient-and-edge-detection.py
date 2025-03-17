import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../images/toba.jpg", 0)
img = cv2.resize(img, (1920, 1080))

lap = cv2.Laplacian(img, cv2.CV_64F, ksize=1)

lap = np.uint8(np.abs(lap))

sobx = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0)
sobx = np.uint8(np.abs(sobx))

soby = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1)
soby = np.uint8(np.abs(soby))

sob = cv2.bitwise_or(sobx, soby)
titles = ["original", "Laplacian", "sobx", "soby", "sob"]
imgs = [img, lap, sobx, soby, sob]

for i in range(len(imgs)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(imgs[i], 'gray')
    plt.xticks([])
    plt.yticks([])

plt.show()
