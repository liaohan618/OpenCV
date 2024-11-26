import numpy as np
import cv2

color = cv2.imread("butterfly.jpg")
#cv2.imshow("Image", color)
#cv2.moveWindow("Image", 0, 0)
print(color.shape)
height, width, channel = color.shape

'1. CHANGE IN COLOR (RGB) : TO RGB BLACK & WHITE'
b, g, r = cv2.split(color)
rgb_split = np.empty([height, width*3, 3], "uint8")
rgb_split[:, 0:width] = cv2.merge([b,b,b])
rgb_split[:, width:width*2] = cv2.merge([g,g,g])
rgb_split[:, width*2:width*3] = cv2.merge([r,r,r])
cv2.imwrite("Channels.jpg", rgb_split)
#cv2.imshow("Channels", rgb_split) # show when run
#cv2.moveWindow("Channels", 0, height)


'2. CHANGE IN RESOLUTION (HSV)'
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(color) # h=HUE, s=saturation, v=value(luminous)
hsv_split = np.concatenate((h,s,v), axis=1) # axis=1 -> images appear side by side
#cv2.imshow("hsv_split", hsv_split)
cv2.imwrite("hsv_split.jpg", hsv_split)


'3. CHANGE IN TRANSPARENCY'
color = cv2.imread("butterfly.jpg")
gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
cv2.imwrite("gray.jpg", gray)
# same as split
b = color[:,:,0]
g = color[:,:,1]
r = color[:,:,2]

rgba = cv2.merge((b,g,r,g)) # last g = only show high green, low green transparent
cv2.imwrite("rgba.jpg", rgba) # save this image in folder when run


'4. GAUSSIAN BLUR'
image = cv2.imread("thresh.jpg")
blur = cv2.GaussianBlur(image,(5,55),0)
cv2.imwrite("Blur.jpg", blur)


'5. DILATION : more light color'
kernel = np.ones((5,5), "uint8")
dilate = cv2.dilate(image, kernel, iterations=1)
cv2.imwrite("Dilation.jpg", dilate)

'6. EROSION : more dark color (erase white)'
kernel = np.ones((5,5), "uint8")
erode = cv2.erode(image, kernel, iterations=1)
cv2.imwrite("Erosion.jpg", erode)


'7. SCALING'
image = cv2.imread("players.jpg", 1)
image_half = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
image_stretch = cv2.resize(image, (600,600))
image_stretch_near = cv2.resize(image, (600,600), interpolation=cv2.INTER_NEAREST)

cv2.imwrite("half.jpg", image_half)
cv2.imwrite("stretch.jpg", image_stretch)
cv2.imwrite("stretch_near.jpg", image_stretch_near)


'8. ROTATING'
M = cv2.getRotationMatrix2D((0,0), -30, 1)
rotate = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imwrite("rotate.jpg", rotate)

cv2.waitKey(0)
cv2.destroyAllWindows()