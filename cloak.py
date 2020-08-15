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

    # Remove noise outside the figure (can tweak shape later)
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

# HSV Upper and Lower bounds config
lower_hsv = None
upper_hsv = None
with open('hsv_param.txt', 'r') as f:
    lower_hsv = f.readline().rstrip()
    lower_hsv = [int(x) for x in lower_hsv.split()]

    upper_hsv = f.readline()
    upper_hsv = [int(x) for x in upper_hsv.split()]

# VideoWriter config
filename = "out.avi"
codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
framerate = 30
resolution = (640, 480)

video_writer = cv2.VideoWriter(filename, codec, framerate, resolution)

# Setup video capture
cap = cv2.VideoCapture(0)

# Wait 5 seconds before taking first frame as empty background
print("Step out of the frame for 5 seconds")
for i in range(5,0,-1):
    time.sleep(1)
    print(i)

# Take an image of the background
_, bg = cap.read()
print("background image taken")

while(True):

    _, img = cap.read()

    # The lower/upper values must be predetermined
    # e.g for a bright green, lower is [39, 74, 7], and upper is [178, 255, 255]
    res, mask, cutout = pipeline(img, bg, lower_hsv, upper_hsv)
    video_writer.write(res)

    # Show all processed layers
    # cv2.imshow('res', res)
    # cv2.imshow('mask', mask)
    cv2.imshow('cutout', cutout)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

video_writer.release()
cap.release()
cv2.destroyAllWindows()
