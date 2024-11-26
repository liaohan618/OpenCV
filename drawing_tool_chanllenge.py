import numpy as np
import cv2

# Global variables
canvas = np.ones([500,500,3],'uint8')*255

radius = 3
color = (0,255,0) # green
pressed = False

# click callback
def click(event, x, y, flags, param):
	global canvas, pressed
	if event == cv2.EVENT_LBUTTONDOWN:
		cv2.circle(canvas, (x,y),radius, color, -1) # -1 means fill
		pressed = True

	# want consecutive circles when dragging = when 1) press button	down & 2) moving mouss
	elif event == cv2.EVENT_MOUSEMOVE and pressed == True:
		cv2.circle(canvas, (x,y), radius, color, -1)

	elif event == cv2.EVENT_LBUTTONUP:
		pressed = False # don't want circles when 1) moving mouse & 2) not pressing

# window initialization and callback assignment
cv2.namedWindow("canvas")
cv2.setMouseCallback("canvas", click)

# Forever draw loop
while True:

	cv2.imshow("canvas",canvas)

	# key capture every 1ms
	ch = cv2.waitKey(1)
	if ch & 0xFF == ord('q'): # when press '1' quit
		break
	elif ch & 0xFF == ord('b'): # when press 'b' change blue
		color = (255, 0, 0)
	elif ch & 0xFF == ord('g'): # when press 'g' change green
		color = (0, 255, 0)
	elif ch & 0xFF == ord('r'): # when press 'r' change red
		color = (0, 0, 255)

cv2.destroyAllWindows()