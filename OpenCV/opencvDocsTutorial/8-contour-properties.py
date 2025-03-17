import numpy as np
import cv2 as cv

img = cv.imread("../images/hexagon.jpg")
img = cv.bilateralFilter(img, 9, 75, 75)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, th1 = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
canny = cv.Canny(th1, 100, 200)
contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cv.drawContours(img, contours, -1, (0, 255, 5), 2)
cv.imshow("Gsa", img)
cv.waitKey(0)
cv.destroyAllWindows()

cnt = contours[1]

x, y, w, h = cv.boundingRect(cnt)
aspect_ratio = float(w) / h
print(aspect_ratio)

area = cv.contourArea(cnt)
react_area = w * h
extent = float(area) / react_area
print(extent)

hull = cv.convexHull(cnt)
hull_area = cv.contourArea(hull)
solidity = float(area) / hull_area
print(solidity)

equi_diameter = np.sqrt(4 * area / np.pi)
print(equi_diameter)

(x, y), (MA, ma), angle = cv.fitEllipse(cnt)
print(angle)

mask = np.zeros(gray.shape, np.uint8)
cv.drawContours(mask, [cnt], 0, 255, -1)
pixelpoints = np.transpose(np.nonzero(mask))
# pixelpoints = cv.findNonzero
white = np.zeros(gray.shape, np.uint8)
white[pixelpoints[:, 0], pixelpoints[:, 1]] = 255
cv.imshow("gas", white)
cv.waitKey(0)
cv.destroyAllWindows()

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(gray, mask=mask)
print(min_val, max_val, min_loc, max_loc)

mean_val = cv.mean(img, mask)
print(mean_val)

leftmost = tuple(cnt[cnt[..., 0].argmin()][0])
rightmost = tuple(cnt[cnt[..., 0].argmax()][0])
topmost = tuple(cnt[cnt[..., 1].argmin()][0])
bottommost = tuple(cnt[cnt[..., 1].argmax()][0])
cv.circle(img, leftmost, 1, (61, 61, 61), 2)
cv.circle(img, rightmost, 1, (61, 61, 61), 2)
cv.circle(img, topmost, 1, (61, 61, 61), 2)
cv.circle(img, bottommost, 1, (61, 61, 61), 2)

cv.imshow("img", img)
cv.waitKey(0)
cv.destroyAllWindows()
