import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("../images/toba.jpg", 0)
hist, bins = np.histogram(img.flatten(), 256, [0, 256])

cdf = hist.cumsum()
cdf_normalized = cdf * float(np.max(hist)) / cdf.max()
plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc="upper left")
plt.show()

cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() * cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype(np.uint8)

img2 = cdf[img]

img = cv.imread("../images/badLight.jpg", 0)

equ = cv.equalizeHist(img)
res = np.hstack((img, equ))
cv.imwrite("../images/badLight2.jpg", res)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)
res = np.hstack((img, cl1))

cv.imwrite("../images/badLight3.jpg", cl1)
