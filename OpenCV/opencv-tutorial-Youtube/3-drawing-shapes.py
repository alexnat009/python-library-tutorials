import numpy as np
import cv2

# img = cv2.imread()
img = np.zeros((680, 680), np.uint8)
# img = cv2.line(img, (0, 0), (100, 100), (255, 0, 0), 2)
# img = cv2.arrowedLine(img, (20, 10), (400, 20), (255, 0, 0), 2)
img = cv2.rectangle(img, (500, 500), (400, 400), (0, 0, 255), 2)
# img = cv2.circle(img, (400, 800), 100, (0, 255, 0), -1)
# img = cv2.putText(img, "toba", (500, 500), cv2.FONT_ITALIC, 4, (255, 255, 255), 10, cv2.LINE_AA)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
