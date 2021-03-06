import cv2
import numpy as np
import os

def nothing(x):
    pass

cv2.namedWindow('image')

cv2.createTrackbar('lh', 'image', 0, 179, nothing)
cv2.createTrackbar('ls', 'image', 0, 255, nothing)
cv2.createTrackbar('lv', 'image', 0, 255, nothing)

cv2.createTrackbar('uh', 'image', 0, 179, nothing)
cv2.createTrackbar('us', 'image', 0, 255, nothing)
cv2.createTrackbar('uv', 'image', 0, 255, nothing)

cap = cv2.VideoCapture(0)

while(True):

    _, img = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Get trackbar positions
    lh = cv2.getTrackbarPos('lh', 'image')
    ls = cv2.getTrackbarPos('ls', 'image')
    lv = cv2.getTrackbarPos('lv', 'image')

    uh = cv2.getTrackbarPos('uh', 'image')
    us = cv2.getTrackbarPos('us', 'image')
    uv = cv2.getTrackbarPos('uv', 'image')

    # define range of blue color in HSV
    lower_colour = np.array([lh, ls, lv])
    upper_colour = np.array([uh, us, uv])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_colour, upper_colour)

    # Remove noise outside the figure
    kernel = np.ones(shape=(5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    inv = cv2.bitwise_not(mask)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)

    #cv2.imshow('frame', frame)
    cv2.imshow('image', mask)
    cv2.imshow('res', res)
    cv2.imshow('inv', inv)

    # Spacebar saves the parameters to a file
    if cv2.waitKey(10) & 0xFF == ord(' '):
        with open('hsv_param.txt', 'w') as f:
            f.write(f"{lh} {ls} {lv} \n")
            f.write(f"{uh} {us} {uv}")

    # q key exits the program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
