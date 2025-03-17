import numpy as np
import cv2 as cv

img = cv.imread("../images/star.png")
img = cv.bilateralFilter(img, 9, 74, 74)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, th1 = cv.threshold(gray, 127, 255, 0)
contours, hierarchy = cv.findContours(th1, 2, 1)
cnt = contours[0]

hull = cv.convexHull(cnt, returnPoints=False)
defects = cv.convexityDefects(cnt, hull)
for i in range(defects.shape[0]):
    start_point, end_point, farthest_point, approx_distance = defects[i, 0]
    start = tuple(cnt[start_point, 0])
    end = tuple(cnt[end_point, 0])
    far = tuple(cnt[farthest_point, 0])
    cv.line(img, start, end, [0, 255, 0], 2)
    cv.circle(img, far, 5, [0, 0, 255], -1)

dist = cv.pointPolygonTest(cnt, (50, 50), True)

print(dist)
cv.imshow("img", img)
cv.waitKey(0)
cv.destroyAllWindows()

img1 = cv.imread("../images/star.png", 0)
img2 = cv.imread("../images/star1.png", 0)

_, th1 = cv.threshold(img1, 127, 255, 0)
_, th2 = cv.threshold(img2, 127, 255, 0)
contours, _ = cv.findContours(th1, 2, 1)
cnt1 = contours[0]
contours, _ = cv.findContours(th1, 2, 1)
cnt2 = contours[0]
ret = cv.matchShapes(cnt1, cnt2, 1, 0.0)
print(ret)


def cartesian_product_simple_transpose(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    return arr.reshape(la, -1).T


img1 = cv.imread("../images/star.png")
gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
_, th1 = cv.threshold(gray, 127, 255, 0)
contours, _ = cv.findContours(th1, 2, 1)
cnt1 = contours[0]

print(img1.shape)
cols, rows = img1.shape[:2]
a = np.arange(0, cols, 1, np.int32)
b = np.arange(0, rows, 1, np.int32)
grid = cartesian_product_simple_transpose(a, b)
print(grid)
for x, y in grid:
    dist = cv.pointPolygonTest(cnt, (int(x), int(y)), True)
    if dist > 50:
        img1[x, y] = [0, 0, 255]
    elif 0 < dist < 50:
        img1[x, y] = [100, 0, 255]
    elif dist == 0:
        img1[x, y] = [255, 255, 255]
    elif -50 < dist < 0:
        img1[x, y] = [255, 0, 0]
    else:
        img1[x, y] = [0, 0, 0]
print(img1)
cv.imshow("ags", img1)
cv.waitKey(0)
cv.destroyAllWindows()
