import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../images/toba.jpg")
img = cv2.resize(img, (1280, 720))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel = 1 / 25 * np.ones((5, 5))

dest = cv2.filter2D(img, 10, kernel)
blur = cv2.blur(img, (5, 5))
gauss = cv2.GaussianBlur(img, (5, 5), 0)
median = cv2.medianBlur(img, 5)
bilateralFilter = cv2.bilateralFilter(img, 9, 75, 75)
titles = ["image", "2D convolution", "blue", "Gblur", "median", "bilat"]
images = [img, dest, blur, gauss, median, bilateralFilter]
for i in range(6):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.xticks([])
    plt.yticks([])
plt.show()
