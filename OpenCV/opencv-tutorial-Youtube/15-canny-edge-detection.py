import numpy as np
import cv2
import matplotlib.pyplot as plt


def nothing(x):
    pass


img = cv2.imread("../images/toba.jpg", 0)
img = cv2.resize(img, (500, 500))
cv2.namedWindow("img")

cv2.createTrackbar("t1", "img", 1, 255, nothing)
cv2.createTrackbar("t2", "img", 1, 255, nothing)
while 1:
    v1 = cv2.getTrackbarPos("t1", "img")
    v2 = cv2.getTrackbarPos("t2", "img")

    cv2.imshow("image", img)
    canny = cv2.Canny(img, v1, v2)
    cv2.imshow("imag2", canny)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
