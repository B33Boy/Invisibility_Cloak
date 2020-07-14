import cv2 as cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_colour = np.array([41, 0, 27])
    upper_colour = np.array([82, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_colour, upper_colour)
    inv = cv2.bitwise_not(mask)

    res = cv2.bitwise_and(frame, frame,mask=inv)

    cv2.imshow('res', res)
    cv2.imshow('mask', mask)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



















