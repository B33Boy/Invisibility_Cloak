import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('image')

cv2.createTrackbar('thresh', 'image', 0, 255, nothing)
cv2.createTrackbar('maxval', 'image', 0, 255, nothing)

cap = cv2.VideoCapture(0)

while(True):
    
    thresh = cv2.getTrackbarPos('thresh', 'image')
    maxval = cv2.getTrackbarPos('maxval', 'image')
    
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grey = cv2.cvtColor(hsv, cv2.COLOR_HSV2GRAY)

    ret, mask = cv2.threshold(grey, thresh, maxval, cv2.THRESH_BINARY)
    cv2.imshow('image', mask)
    cv2.imshow('hsv', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
