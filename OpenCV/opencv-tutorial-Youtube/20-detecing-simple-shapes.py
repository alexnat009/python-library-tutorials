import numpy as np
import cv2


def nothing(x):
    print(x)


img = cv2.imread("../images/shapes.jpg")
img = cv2.resize(img, (500, 500))
img = cv2.GaussianBlur(img, (3, 3), 6)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, th1 = cv2.threshold(imgGray, 230, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    cv2.drawContours(img, approx, -1, (0, 0, 255), 6)
    x, y = approx.ravel()[[0, 1]]
    if len(approx) == 3:
        cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    elif len(approx) == 4:
        x, y, h, w = cv2.boundingRect(approx)
        aspectR = float(w) / h
        if 0.95 <= aspectR <= 1.05:
            cv2.putText(img, "Square", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        else:
            cv2.putText(img, "Rectangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    elif len(approx) == 5:
        cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    elif len(approx) == 6:
        cv2.putText(img, "Hexagon", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    else:
        cv2.putText(img, "circle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
cv2.imshow('t', img)
cv2.imshow("t2", th1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#
# cv2.namedWindow("track")
# cv2.createTrackbar("t1", "track", 0, 255, nothing)
# cv2.createTrackbar("t2", "track", 0, 255, nothing)
# while 1:
#     cv2.imshow('t', img)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break
#     t1 = cv2.getTrackbarPos("t1", "track")
#     t2 = cv2.getTrackbarPos("t2", "track")
#     _, th1 = cv2.threshold(imgGray, t1, t2, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     for contour in contours:
#         approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
#         cv2.drawContours(img, approx, 0, (0, 0, 255), 2)
#         x, y = approx.ravel()[[0, 1]]
#         if len(approx) == 3:
#             cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#         elif len(approx) == 4:
#             x, y, h, w = cv2.boundingRect(approx)
#             aspectR = float(w) / h
#             if 0.95 <= aspectR <= 1.05:
#                 cv2.putText(img, "Square", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#             else:
#                 cv2.putText(img, "Rectangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#         elif len(approx) == 5:
#             cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#         elif len(approx) == 6:
#             cv2.putText(img, "Hexagon", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#         else:
#             cv2.putText(img, "circle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#
#     cv2.imshow("t2", th1)
#
# cv2.destroyAllWindows()
#
