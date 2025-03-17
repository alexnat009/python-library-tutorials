import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((200, 200), np.uint8)
cv2.rectangle(img, (0, 100), (200, 200), (255), -1)
cv2.rectangle(img, (0, 50), (100, 100), (127), -1)
# gg = cv2.imread("../images/gradient.jpg")
cv2.imshow("gsad", img)

plt.hist(img.ravel(), 256, [0, 256])
plt.show()
k = cv2.waitKey(1) & 0xFF
if k == 27:
    cv2.destroyAllWindows()
