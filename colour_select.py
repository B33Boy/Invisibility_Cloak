import cv2 as cv2
import numpy as np

cap = cv2.VideoCapture(0)

def nothing(x):
    pass

cv2.namedWindow('image')

cv2.createTrackbar('lb', 'image', 0, 255, nothing)
cv2.createTrackbar('lg', 'image', 0, 255, nothing)
cv2.createTrackbar('lr', 'image', 0, 255, nothing)

cv2.createTrackbar('ub', 'image', 0, 255, nothing)
cv2.createTrackbar('ug', 'image', 0, 255, nothing)
cv2.createTrackbar('ur', 'image', 0, 255, nothing)


while(True):
    # Take each frame
    _, frame = cap.read()
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Get trackbar positions
    lb = cv2.getTrackbarPos('lb', 'image')
    lg = cv2.getTrackbarPos('lg', 'image')
    lr = cv2.getTrackbarPos('lr', 'image')
    
    ub = cv2.getTrackbarPos('ub', 'image')
    ug = cv2.getTrackbarPos('ug', 'image')
    ur = cv2.getTrackbarPos('ur', 'image')
 
    # define range of blue color in HSV
    lower_colour = np.array([lb, lg, lr])
    upper_colour = np.array([ub, ug, ur])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_colour, upper_colour)
    inv = cv2.bitwise_not(mask) 
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    #cv2.imshow('frame', frame)
    cv2.imshow('image', mask)
    cv2.imshow('res', res)
    cv2.imshow('inv', inv)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
