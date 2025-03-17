import numpy as np
import cv2


def nothing(x):
    print(x)


img = cv2.imread("../images/color.jpg", 0)
cv2.namedWindow("thresh")
bar = cv2.createTrackbar("T", "thresh", 0, 255, nothing)
while 1:
    cv2.imshow("gradient", img)
    t = cv2.getTrackbarPos("T", "thresh")
    _, th1 = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
    _, th2 = cv2.threshold(img, t, 255, cv2.THRESH_BINARY_INV)
    _, th3 = cv2.threshold(img, t, 255, cv2.THRESH_TRUNC)
    _, th4 = cv2.threshold(img, t, 255, cv2.THRESH_TOZERO)
    _, th5 = cv2.threshold(img, t, 255, cv2.THRESH_TOZERO_INV)
    cv2.imshow("thresholding1", th1)
    cv2.imshow("thresholding2", th2)
    cv2.imshow("thresholding3", th3)
    cv2.imshow("thresholding4", th4)
    cv2.imshow("thresholding5", th5)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
