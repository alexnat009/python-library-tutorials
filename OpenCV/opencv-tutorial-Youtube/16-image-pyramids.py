import numpy as np
import cv2

img = cv2.imread("../images/toba.jpg", 0)
img = cv2.resize(img, (500, 500))
pyrUp = cv2.pyrUp(img)
pyrDown = cv2.pyrDown(img)
cv2.imshow("original", img)
cv2.imshow("done", pyrDown)
# cv2.imshow("up", pyrUp)
cv2.waitKey(0)
cv2.destroyAllWindows()
