import numpy as np
import cv2


def nothing(x):
    pass


img = cv2.imread("images/sudoku.jpg", 0)
img = cv2.resize(img, (500, 500))
cv2.namedWindow("thresh")
cv2.createTrackbar("Tv", "thresh", 1, 255, nothing)
cv2.createTrackbar("Bv", "thresh", 3, 255, nothing)
cv2.createTrackbar("Cv", "thresh", 0, 255, nothing)

while 1:
    cv2.imshow("image", img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    t = cv2.getTrackbarPos("Tv", "thresh")
    b = cv2.getTrackbarPos("Bv", "thresh")
    c = cv2.getTrackbarPos("Cv", "thresh")
    if b % 2 == 0:
        b += 3
    _, th1 = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
    print(b)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, b, c)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b, c)

    # cv2.imshow("thresh1", th1)
    cv2.imshow("thresh2", th2)
    cv2.imshow("thresh3", th3)

cv2.destroyAllWindows()
