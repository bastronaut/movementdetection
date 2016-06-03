from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import matplotlib

def backgroundsubtractor(image):
    fgmask = fgbg.apply(image)
    cv2.imshow('image',fgmask)
    return fgmask


def pedestriandetection(image):
	# detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
    	padding=(8, 8), scale=1.05)
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    return image


if __name__ == "__main__":
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # initialize foreground/background detector
    fgbg = cv2.createBackgroundSubtractorMOG2()
    # set the video capturing
    cap = cv2.VideoCapture(0)

    pedestrianresults = []
    while len(pedestrianresults)  < 5:
        ret, imagecap = cap.read()
        image = imutils.resize(imagecap, width=min(400, imagecap.shape[1]))
        # orig = image.copy()

        # fgbgimage = backgroundsubtractor(image)
        pedimage = pedestriandetection(image)

        cv2.imshow('image', fgbgimage)

        pedestrianresults.append(pedimage)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
