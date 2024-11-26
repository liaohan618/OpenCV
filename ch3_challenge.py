import cv2
import numpy as np
from random import randint

img = cv2.imread('fuzzy.png', 1)

# gray
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# blur
blur = cv2.GaussianBlur(gray, (3,3), 0)

# adaptive threshold
thres = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 205, 1)
cv2.imshow('thres', thres)

# contours
contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

# area >= 1000
filtered = []
for c in contours :
    area = cv2.contourArea[c]
    if area < 1000:
        continue
    filtered.append(c)

# draw contours
objects = np.zeros([img.shape[0], img.shape[1], 3], 'uint8')

for c in filtered:
    col = (randint(0,255), randint(0,255), randint(0,255))
    cv2.drawContours(objects, [c], -1, col, -1)
    area = cv2.contourArea(c)
    print(area)

cv2.imshow("final", objects)








cv2.waitKey(0)
cv2.destroyAllWindows()