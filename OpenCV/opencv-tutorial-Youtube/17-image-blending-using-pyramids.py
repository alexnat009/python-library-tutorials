import numpy as np
import cv2

gradient = cv2.imread("../images/color.jpg")
toba = cv2.imread("../images/toba.jpg")

toba = cv2.resize(toba, (500, 500))
gradient = cv2.resize(gradient, (500, 500))

combo = np.hstack((toba[:, :255], gradient[:, 255:]))

# cv2.imshow("imag1", color)
# cv2.imshow("imag2", gradient)


# //////////////////
toba_copy = toba.copy()
gp_toba = [toba_copy]

for i in range(6):
    toba_copy = cv2.pyrDown(toba_copy)
    gp_toba.append(toba_copy)


toba_copy = gp_toba[5]
lp_toba = [toba_copy]
for i in range(5, 0, -1):
    gaussian_expanded = cv2.pyrUp(gp_toba[i])
    lap = cv2.subtract(gp_toba[i - 1], gaussian_expanded)
    lp_toba.append(lap)

#     ////////////
gradient_copy = gradient.copy()
gp_gradient = [gradient_copy]


for i in range(6):
    gradient_copy = cv2.pyrDown(gradient_copy)
    gp_gradient.append(gradient_copy)


gradient_copy = gp_gradient[5]
lp_gradient = [gradient_copy]
for i in range(5, 0, -1):
    gaussian_expanded = cv2.pyrUp(lp_gradient_copy[i])
    lap = cv2.subtract(lp_gradient_copy[i - 1], gaussian_expanded)
    lp_gradient.append(lap)

result = []
n = 0
for lap1, lap2 in zip(lp_color1, lp_color2):
    n += 1
    cols, rows, ch = lap1.shape
    lap = np.hstack((lap1[:, 0:int(cols / 2)], lap2[:, int(cols / 2):]))
    result.append(lap)

reconstruct = result[0]
for i in range(1, 6):
    reconstruct = cv2.pyrUp(reconstruct)
    reconstruct = cv2.add(result[i], reconstruct)

cv2.imshow("fsdg", reconstruct)

cv2.imshow("imag3", combo)
cv2.waitKey(0)
cv2.destroyAllWindows()
