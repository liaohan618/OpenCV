import numpy as np
import cv2

# GOAL: Detect eyes from faces using Haar cascade method

# 0. Import the image
img = cv2.imread("faces.jpeg", 1)

# 1. Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Load Haar cascade for eyes
path = 'haarcascade_eye.xml'
eye_cascade = cv2.CascadeClassifier(path)

# 3. Detect eyes
eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(70, 70), maxSize=(100,100))
print(f"Number of eyes detected: {len(eyes)}")

# 4. Draw circles around detected eyes
for (x, y, w, h) in eyes:
    center = (x + w // 2, y + h // 2)  # Center of the eye
    radius = w // 2  # Approximate radius
    cv2.circle(img, center, radius, (255, 0, 0), 3)

# 5. Display the image with detected eyes
cv2.imwrite("Eyes_Detection.jpg", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
