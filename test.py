import cv2
import numpy as np

'Example 1'
template = cv2.imread('train1.png', 0)
frame = cv2.imread('test1.png', 0)
result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

# Find the best match location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print(f"Max value: {max_val}, Location: {max_loc}")

# Draw at the location of the match
top_left = max_loc
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0]) # size of original image
cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

cv2.imwrite("matching.jpg", frame)
cv2.imshow("Matching Result", frame)


'Example 2'
template = cv2.imread('train1.png', 0)
frame = cv2.imread('test2.png', 0)
result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

# Find the best match location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print(f"Max value: {max_val}, Location: {max_loc}")

# Draw 
top_left = max_loc
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0]) # size of original image
cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

cv2.imwrite("matching2.jpg", frame)
cv2.imshow("Matching Result2", frame)


'Example 3'
template = cv2.imread('train1.png', 0)
frame = cv2.imread('test3.png', 0)
result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

# Find the best match location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print(f"Max value: {max_val}, Location: {max_loc}")

# Draw 
top_left = max_loc
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0]) # size of original image
cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

cv2.imwrite("matching3.jpg", frame)
cv2.imshow("Matching Result3", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()