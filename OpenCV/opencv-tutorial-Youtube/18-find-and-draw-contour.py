import numpy as np
import cv2

img = cv2.imread("../images/toba.jpg")
img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.resize(img, (600, 600))
_, th = cv2.threshold(grayImg, 100, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
print(f'Number of contours = {len(contours)}')
# contours = contours[1:]
cv2.drawContours(img, contours, -1, (10, 255, 10), 1)
cv2.imshow("img1", img)
cv2.imshow("img2", th)
cv2.waitKey(0)
cv2.destroyAllWindows()
