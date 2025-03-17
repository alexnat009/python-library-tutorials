import numpy as np
import cv2

img = cv2.imread("../images/sudoku.jpg")
img = cv2.resize(img, (500, 500))
# img = cv2.GaussianBlur(img, (5, 5), 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 200, 200)
cv2.imshow("edges", edges)
lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 100, minLineLength=10, maxLineGap=9)
for line in lines:
    print(line)
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
cv2.imshow("img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
