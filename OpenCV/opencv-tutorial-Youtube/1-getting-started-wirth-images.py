import cv2 as cv
import numpy as np



def nothing(x):
    print(x)


img = np.zeros((500, 500, 3))
cv.namedWindow("imggg")
cv.createTrackbar("t1", "imggg", 0, 100, nothing)
x = np.arange(0, 100, 1)
print(x)

for i in x:
    t1 = cv.getTrackbarPos("t1", "imggg")
    print(i)
    if i < 100:
        cv.circle(img, (100, i), 50, (255, 0, 0), 2)
    else:
        i = 100
        cv.circle(img, (100, i), 50, (255, 0, 0), 2)

    cv.imshow("img", img)
    cv.waitKey(0)

cv.destroyAllWindows()
