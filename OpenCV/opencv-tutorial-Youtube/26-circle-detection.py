import numpy as np
import cv2

img = cv2.imread("../images/shapes.jpg")
img = cv2.resize(img, (1400, 700))
output = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(image=gray,
                           method=cv2.HOUGH_GRADIENT,
                           dp=1,
                           minDist=20,
                           param1=50,
                           param2=60,
                           minRadius=0,
                           maxRadius=0)
detected_circles = np.uint16(np.around(circles))
for (x, y, r) in detected_circles[0, :]:
    cv2.circle(output, (x, y), r, (0, 255, 0), 2)
    cv2.circle(output, (x, y), 2, (0, 255, 255), 2)

cv2.imshow("img", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
