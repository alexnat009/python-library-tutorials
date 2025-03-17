import numpy as np
import cv2 as cv

img = cv.imread("../images/irectange.png")
img = cv.bilateralFilter(img, 9, 75, 75)
img1 = img.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

_, th1 = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(th1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

cv.drawContours(img, contours, -1, (0, 0, 255), 3)
cnt = contours[1]
M = cv.moments(cnt)
print(M)
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
print(cx, cy)
area = cv.contourArea(cnt)
perimeter1 = cv.arcLength(cnt, False)
perimeter2 = cv.arcLength(cnt, True)
print(f'area is {area}')
print(f'closed perimeter {perimeter2}')
cv.circle(img, (cx, cy), 3, (255, 0, 0), 3)
cv.imshow("igm", img)

for c in contours:
    epsilon = 0.05 * cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, epsilon, True)
    cv.drawContours(img1, [approx], 0, (0, 255, 0), 2)

hull = cv.convexHull(cnt)
print(hull)

k = cv.isContourConvex(cnt)
print(k)
cv.imshow("ga", img1)
cv.waitKey(0)
cv.destroyAllWindows()

img = cv.imread("../images/fla.png")
img = cv.bilateralFilter(img, 9, 75, 75)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, th1 = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(th1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cv.drawContours(img, contours, -1, (0, 0, 255), 3)

cnt = contours[0]

x, y, w, h = cv.boundingRect(cnt)
cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int0(box)
cv.drawContours(img, [box], 0, (0, 51, 255), 2)

(x, y), radius = cv.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)

cv.circle(img, center, radius, (100, 0, 255), 2)

ellipse = cv.fitEllipse(cnt)
cv.ellipse(img, ellipse, (0, 255, 0), 2)

rows, cols = img.shape[:2]
[vx, vy, x0, y0] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x0 * vy / vx + y0))
righty = int(((cols - x0) * vy / vx) + y0)
cv.line(img, (cols - 1, righty), (0, lefty), (0, 100, 100), 2)


cv.imshow("img", img)
cv.waitKey(0)
cv.destroyAllWindows()
