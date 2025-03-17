import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
img = cv2.imread("../images/toba.jpg", 0)

img = cv2.resize(img, (1280, 720))
cv2.imshow("image", img)
_, th1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
_, th2 = cv2.threshold(img, 50, 255, cv2.THRESH_TRUNC)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)
th4 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
titles = ["Original", "BINARY", "TRUNC", "ADAPTIVE_MEAN", "ADAPTIVE_GAUSSIAN"]
images = [img, th1, th2, th3, th4]
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
for i in range(5):
	plt.subplot(2, 3, i + 1)
	plt.imshow(images[i], 'gray')
	plt.title(titles[i])
	plt.xticks([])
	plt.yticks([])

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
