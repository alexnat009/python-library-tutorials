import numpy as np
import cv2 as cv

img = cv.imread("../images/toba.jpg", 0)

img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

laplacian = cv.Laplacian(img, cv.CV_64F)
laplacian = np.uint8(np.abs(laplacian))
sobelx = cv.Sobel(img, cv.CV_64F, dx=1, dy=0, ksize=1)
sobelx = np.uint8(np.abs(sobelx))
sobely = cv.Sobel(img, cv.CV_64F, dx=0, dy=1, ksize=1)
sobely = np.uint8(np.abs(sobely))
sobel = cv.bitwise_or(sobely, sobelx)

canny = cv.Canny(img, 100, 200)
cv.imshow("original", img)
cv.imshow("laplacian", laplacian)
cv.imshow("sobelx", sobelx)
cv.imshow("sobely", sobely)
cv.imshow("sobel", sobel)
cv.imshow("canny", canny)

cv.waitKey(0)
cv.destroyAllWindows()
