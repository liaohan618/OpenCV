import numpy as np
import cv2

'1. CHANGE COLOR TO BINARY'
bw = cv2.imread('/Users/liaohanwang/Downloads/opencv/ch3 Object Detection/detect_blob.png', 0)  # 0 = grayscale
height, width = bw.shape[0:2]

# Create an empty binary image
binary = np.zeros([height, width], dtype='uint8')
threshold = 85 # if color number greater than it then black (1), otherwise white (0)

for row in range(height):
    for col in range(width):
        if bw[row, col] > threshold:
            binary[row, col] = 255  # Set pixel to white

cv2.imwrite('binary.png', binary)


'2. ADAPTIVE BINARY'
img = cv2.imread('/Users/liaohanwang/Downloads/opencv/ch3 Object Detection/sudoku.png', 0)
#cv2.imshow('Origin', img)

# basic
ret, threshold_basic = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
cv2.imwrite('binary_basic.png', threshold_basic) # problem : shade also become black that made binary image not ideal

# adaptive
thres_adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imwrite('adaptive.png', thres_adapt)


'3. CHANGE HSV'
img = cv2.imread('faces.jpeg', 1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]
hsv_split = np.concatenate((h,s,v), axis = 1) # 1 = horizontally
cv2.imwrite('split_hsv.jpeg', hsv_split)

# saturation filter
ret, min_sat = cv2.threshold(s, 40, 255, cv2.THRESH_BINARY) # saturation > 40 will be white
cv2.imwrite('sat_filter.jpeg', min_sat)

# hue filter
ret, max_hue = cv2.threshold(h, 15, 255, cv2.THRESH_BINARY_INV) # inverse - hue < 15 will be white
cv2.imwrite('hue_filter.jpeg', max_hue)

# combine filters
final = cv2.bitwise_and(min_sat, max_hue)
cv2.imwrite('final.jpeg', final)


'4. Contours'
img = cv2.imread('detect_blob.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

#cv2.imshow('binary', threshold)
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # CHAIN_APPROX_SIMPLE = only want key contours (vertex)

img2 = img.copy() # modify img2 will NOT change img
index = -1 # -1 = all contours
thickness = 4
color = (255, 0, 255) # pink
cv2.drawContours(img2, contours, index, color, thickness)
cv2.imwrite('contour.png', img2)


'5. Area & Perimeter & Centroid'
object = np.zeros([img.shape[0], img.shape[1], 3], 'uint8')
for c in contours :
    cv2.drawContours(object, [c], -1, color, -1)
    area = cv2.contourArea(c) # area = number of pixels enclosed in contour
    prem = cv2.arcLength(c, True)
    # centroid
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cv2.circle(object, (cx, cy), 4, (0, 255, 255), -1)
    print(f"Area : {area}, Perimeter : {prem}")

cv2.imwrite("Contour.png", object)


'6. Canny Edge Detection'
img = cv2.imread('tomatoes.jpg', 1)

# problem : regular hsv detect couldn't show number of tomatoes clearly
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
res, threshold = cv2.threshold(hsv[:, :, 0], 25, 255, cv2.THRESH_BINARY_INV) # extract color with hue < 25
#cv2.imshow("test", threshold)

# canny edge detect
edges = cv2.Canny(img, 100, 200, apertureSize=3)
cv2.imwrite("canny.jpg", edges)

'7. ultimate combined canny edge detect : Detect number of tomatoes'
edge_inv = 255 - edges # canny edges are shown as black insted of white

# Erosion to increase size of border
kernel = np.ones((3,3), "uint8")
erode = cv2.erode(edge_inv, kernel, iterations=1) 

# Canny Edge
canny_thres = cv2.bitwise_and(erode, threshold)
#cv2.imshow("canny_thres.jpg", canny_thres)

contours, hierarchy = cv2.findContours(canny_thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
object = img.copy()

# Area
for c in contours :
    area = cv2.contourArea(c)
    if area < 300:
        continue # area too small so neglect
    print("Area : ", area)

    cv2.drawContours(object, [c], -1, (255, 255, 255), 1)

    # Centroid
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cv2.circle(object, (cx, cy), 4, (255, 255, 0), -1)

cv2.imwrite("FINAL.jpg", object)



cv2.waitKey(0)
cv2.destroyAllWindows()
