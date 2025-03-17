import numpy as np
import cv2

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (480, 640))

fgbg = cv2.createBackgroundSubtractorKNN()
while 1:
    ret, frame = cap.read()
    if frame is None:
        break
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # print(fgmask.shape)
    out.write(fgmask)
    cv2.imshow("gsa", fgmask)
    # cv2.imshow("fg gsa", fgmask)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
