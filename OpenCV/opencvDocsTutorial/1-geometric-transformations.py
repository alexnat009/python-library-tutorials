import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

img = cv.imread("../images/toba.jpg", 0)
img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
# cv.imshow("fas", img)

# # Scaling
# # cv.INTER_CUBIC & cv.INTER_LINEAR for zooming
# # cv.INTER_AREA for shrinking
# res1 = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
# cv.imshow("res1", res1)
# res2 = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
# cv.imshow("res2", res2)
# cv.waitKey(0)

# Translation

# rows, cols = img.shape

# M = np.float32([1, 0, 100, 0, 1, 50]).reshape((2, 3))
# dst = cv.warpAffine(img, M, (cols, rows))
#
# cv.imshow("translated image", dst)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Rotation
# rows, cols = img.shape
# M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 50, 1)
# dst = cv.warpAffine(img, M, (cols, rows))
# cv.imshow("rotated image", dst)
# cv.waitKey(0)
# cv.destroyAllWindows()


# Affine transformation
img = cv.imread("../images/toba.jpg")
rows, cols, ch = img.shape
#
# pts1 = np.float32([50, 50, 200, 50, 50, 200]).reshape(3, 2)
# pts2 = np.float32([10, 100, 200, 50, 100, 250]).reshape(3, 2)
#
# M = cv.getAffineTransform(pts1, pts2)
#
# dst = cv.warpAffine(img, M, (cols, rows))
#
# plt.subplot(121)
# plt.imshow(img)
# plt.title("Input")
#
# plt.subplot(122)
# plt.imshow(dst)
# plt.title("Output")
#
# plt.show()

# Perspective transformation

pts1 = np.float32([56, 65, 600, 52, 28, 607, 570, 490]).reshape(4, 2)
pts2 = np.float32([0, 0, 700, 0, 0, 700, 700, 700]).reshape(4, 2)

M = cv.getPerspectiveTransform(pts1, pts2)

dst = cv.warpPerspective(img, M, (700, 700))

plt.subplot(121)
plt.imshow(img)
plt.title('Input')

plt.subplot(122)
plt.imshow(dst)
plt.title('Output')
plt.show()
