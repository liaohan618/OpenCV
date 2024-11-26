import numpy as np
import cv2

# algorithm : search over all the pixels of frame, compare template to find match
# restrictions : same rotation, size

'1. FEATURE DETECTION'
template = cv2.imread("template.jpg", 0) # a soccer
frame = cv2.imread("players.jpg", 0) # a photo that contains soccer somewhere
result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
#cv2.imshow('result', result)

# locate the matching result
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) 
print(max_val, max_loc)

# Scale the result to the range 0-255 for visualization
result_scaled = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.circle(result_scaled, max_loc, 15, 255, 2) # circles the matching pixel
cv2.imwrite("matching.jpg", result_scaled)


'2. FACE DETECTION (HAAR CASCADE)'
img = cv2.imread("faces.jpeg", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
path = 'haarcascade_frontalface_default.xml' # pre-trained for face detection
face_cascade = cv2.CascadeClassifier(path)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(40,40))
print(len(faces))

# locate the matching result
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
cv2.imwrite("Face.jpg", img)


cv2.waitKey(0)
cv2.destroyAllWindows()