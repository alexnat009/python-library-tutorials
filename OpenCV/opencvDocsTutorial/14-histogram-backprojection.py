import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# # roi is the object or region of object we need to find
roi = cv.imread("../images/tobaWater.png")
hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

# # target is the image we search in
target = cv.imread("../images/toba.jpg")
hsvt = cv.cvtColor(target, cv.COLOR_BGR2HSV)

# numpy algorithm

# # Find the histograms using calcHIst. Can be done witg np.histogram2d also
# M = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
# I = cv.calcHist([hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])
#
# h, s, v = cv.split(hsvt)
# R = np.empty(M.shape)
# np.divide(M, I, out=R, where=I != 0)
# B = R[h.ravel(), s.ravel()]
# B = np.minimum(B, 1).reshape(hsvt.shape[:2])
#
# # now apply a convolution with a circular disc B = D* B where D is the disc kernel
#
# disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
# cv.filter2D(B, -1, disc, B)
# B = np.uint8(B)
# cv.normalize(B, B, 0, 255, cv.NORM_MINMAX)
#
# # Now the location of maximum intensity gives us the location of object
# ret, thresh = cv.threshold(B, 50, 255, 0)
# thresh = cv.merge((thresh, thresh, thresh))
# res = cv.bitwise_and(target, thresh)
# res = np.vstack((target, thresh, res))
# cv.imwrite('../images/res.jpg', res)

# OpenCV algorithm
# calculate object histogram
roiHist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

# normalize histogram and apply backProjection
cv.normalize(roiHist, roiHist, 0, 255, cv.NORM_MINMAX)
dst = cv.calcBackProject([hsvt], [0, 1], roiHist, [0, 180, 0, 256], 1)

# Now convolve with circular dist
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
cv.filter2D(dst, -1, disc, dst)

# threshold and binary AND
ret, thresh = cv.threshold(dst, 50, 255, 0)
thresh = cv.merge((thresh, thresh, thresh))
res = cv.bitwise_and(target, thresh)

res = np.vstack((target, thresh, res))
cv.imwrite("../images/res2.jpg", res)
