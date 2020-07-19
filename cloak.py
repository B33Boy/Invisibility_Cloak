import cv2 as cv2
import numpy as np
import time

def pipeline(img, img_bg, range_lower, range_upper):

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_colour = np.array(range_lower)
    upper_colour = np.array(range_upper)

    # Get the mask based
    mask = cv2.inRange(hsv, lower_colour, upper_colour)

    # Remove noise outside the figure
    kernel = np.ones(shape=(5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # uncomment the line below for testing
    # cv2.imshow('morph_open', open)

    # Get the inverse mask such that roi is black
    inv = cv2.bitwise_not(mask)

    # Original image with area of interest shaded with black
    res = cv2.bitwise_and(img, img, mask=inv)

    # Get background portion with mask shape
    cutout = cv2.bitwise_and(bg, bg, mask=mask)

    # add the images
    res = cv2.add(res, cutout)

    return res, mask, cutout


cap = cv2.VideoCapture(0)

# Wait 5 seconds before taking first frame as empty background
time.sleep(5)

_, bg = cap.read()
print("background image taken")

while(True):

    _, img = cap.read()

    # The lower/upper values must be predetermined
    res, mask, cutout = pipeline(img, bg, [158, 48, 79], [255, 255, 255])

    cv2.imshow('res', res)
    cv2.imshow('mask', mask)
    cv2.imshow('cutout', cutout)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
