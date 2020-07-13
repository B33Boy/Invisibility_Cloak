import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('image')

cv2.createTrackbar('thresh', 'image', 0, 255, nothing)
cv2.createTrackbar('maxval', 'image', 0, 255, nothing)

img = cv2.imread('logo.png')
img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

while(True):

    thresh = cv2.getTrackbarPos('thresh', 'image')
    maxval = cv2.getTrackbarPos('maxval', 'image')

    ret, mask = cv2.threshold(img2gray, thresh, maxval, cv2.THRESH_BINARY)
    cv2.imshow('image', mask)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
