import cv2
import numpy as np
# import argparse

def pipeline(img, img_bg, range_lower, range_upper):

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_colour = np.array(range_lower)
    upper_colour = np.array(range_upper)

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_colour, upper_colour)

    # Remove noise outside the figure
    kernel = np.ones(shape=(5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('morph_open', open)

    inv = cv2.bitwise_not(mask)

    # Original image with area of interest shaded with black
    res = cv2.bitwise_and(img, img, mask=inv)

    # Get background portion with mask shape
    cutout = cv2.bitwise_and(img_bg, img_bg, mask=mask)

    res = cv2.add(res, cutout)

    return res, mask, cutout

face_cascade = cv2.CascadeClassifier('resource/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('john_cena.mp4')

img_bg = cv2.imread('john_cena_bg.jpg')

while(cap.isOpened()):

    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.8, 1)
    # for (x,y,w,h) in faces:
    #     img = cv2.rectangle(img, (x,y), (x-15+int(w*1.2),y-15+int(h*1.2)), (255,0,0), 2)
    #     roi_color = img[y:y+h, x:x+w]

    res, mask, cutout = pipeline(img, img_bg, [0, 7, 15], [61, 94, 255])

    cv2.imshow('roi_color', res)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
