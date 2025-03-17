import numpy as np
import cv2 as cv

img = cv.imread("../images/shapes.jpg")
img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
img = cv.bilateralFilter(img, 9, 75, 75)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

th1 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 3)

# th1 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 3)
contours, hierarchy = cv.findContours(th1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))
# print(contours[4])
# print(hierarchy)
cv.drawContours(img, contours, -1, (8, 100, 123), 3)

cv.imshow("img", img)
cv.waitKey(0)
cv.destroyAllWindows()
