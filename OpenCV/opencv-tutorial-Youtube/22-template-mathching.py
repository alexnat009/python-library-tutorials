import cv2
import numpy as np

img = cv2.imread("../images/toba.jpg")
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.resize(img, (700, 700))

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
