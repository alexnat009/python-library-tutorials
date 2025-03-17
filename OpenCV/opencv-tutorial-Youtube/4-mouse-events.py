import numpy as np
import cv2
import matplotlib.pyplot as plt


# events = [i for i in dir(cv2) if 'EVENT' in i]
# print(*events)

def click_event(event, x, y, flags, param):
    font = cv2.FONT_ITALIC
    if event == cv2.EVENT_LBUTTONDOWN:
        text = f'x={x};y={y}'
        print(text)
        cv2.putText(img, text, (x, y), font, 0.4, (255, 255, 0), 2)
        cv2.imshow("image", img)
    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        text = f'{blue};{green};{red}'
        print(text)
        cv2.putText(img, text, (x, y), font, 0.4, (0, 255, 255), 2)
        cv2.imshow("image", img)


def click_event2(event, x, y, flags, param):
    font = cv2.FONT_ITALIC
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        points.append((x, y))
        if len(points) >= 2:
            cv2.line(img, points[-1], points[-2], (255, 0, 0), 5)
        cv2.imshow("image", img)


def click_event3(event, x, y, flags, param):
    font = cv2.FONT_ITALIC
    if event == cv2.EVENT_LBUTTONDOWN:
        blue, green, red = img[y, x]
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        myColorImage = np.zeros((512, 512, 3), np.uint8)
        myColorImage[:] = [blue, green, red]

        cv2.imshow("color", myColorImage)


# img = np.zeros((512, 512, 3), np.uint8)
img = cv2.imread("toba.jpg", 1)
points = []
cv2.imshow("image", img)
cv2.setMouseCallback("image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

