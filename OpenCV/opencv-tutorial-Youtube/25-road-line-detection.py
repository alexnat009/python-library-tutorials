import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("road.jpg")
img = cv2.GaussianBlur(img, (3, 3), 3)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

width = img.shape[1]
height = img.shape[0]
region_if_interest_vertices = [
    [0, height],
    [width / 2, height / 2],
    [830, 360],
    [width, height]
]


def region_of_interest(img, vertices):
    print(vertices)
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, np.int32([vertices]), match_mask_color)
    return cv2.bitwise_and(img, mask)


def draw_lines(img, lines):
    img = img.copy()
    blank = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return cv2.addWeighted(img, 0.8, blank, 1, 0.0)


gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
canny = cv2.Canny(gray, 100, 200)
cropped_image = region_of_interest(canny, region_if_interest_vertices)
lines = cv2.HoughLinesP(image=cropped_image, rho=1, theta=np.pi / 180, threshold=100, minLineLength=10, maxLineGap=60)

image_with_lines = draw_lines(img, lines)
plt.imshow(image_with_lines, 'gray')
plt.show()
