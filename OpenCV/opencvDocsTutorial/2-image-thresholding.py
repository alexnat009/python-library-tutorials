import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../images/sudoku.jpg", 0)
img = cv2.resize(img, (500, 500))

ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret, thresh3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("THRESH_BINARY", thresh1)
cv2.imshow("THRESH_OTSU", thresh2)
cv2.imshow("THRESH_OTSUwBlur", thresh3)
cv2.waitKey(0)
cv2.destroyAllWindows()
# def nothing(x):
#     pass
#
#
# cv2.namedWindow("imga")
# cv2.createTrackbar("t1", "imga", 0, 255, nothing)
# cv2.createTrackbar("maxvalue", "imga", 0, 255, nothing)
#
# while 1:
#     cv2.imshow("original", img)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
#     t1 = cv2.getTrackbarPos("t1", "imga")
#     maxvalue = cv2.getTrackbarPos("maxvalue", "imga")
#
#     ret, thresh1 = cv2.threshold(img, t1, maxvalue, cv2.THRESH_BINARY)
#     ret, thresh2 = cv2.threshold(img, t1, maxvalue, cv2.THRESH_BINARY_INV)
#     ret, thresh3 = cv2.threshold(img, t1, maxvalue, cv2.THRESH_TRUNC)
#     ret, thresh4 = cv2.threshold(img, t1, maxvalue, cv2.THRESH_TOZERO)
#     ret, thresh5 = cv2.threshold(img, t1, maxvalue, cv2.THRESH_TOZERO_INV)
#     cv2.imshow("THRESH_BINARY", thresh1)
#     cv2.imshow("THRESH_BINARY_INV", thresh2)
#     cv2.imshow("THRESH_TRUNC", thresh3)
#     cv2.imshow("THRESH_TOZERO", thresh4)
#     cv2.imshow("THRESH_TOZERO_INV", thresh5)
#     th1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#     th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
#     cv2.imshow("ADAPTIVE_THRESH_GAUSSIAN_C", th1)
#     cv2.imshow("ADAPTIVE_THRESH_MEAN_C", th2)
#
# cv2.destroyAllWindows()
