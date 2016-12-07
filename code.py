#!/usr/bin/python2
# -*- coding: utf-8 -*-

import sys, os, cv2
import scipy as sp
import scipy.misc as spmisc
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

def main(argv):
    '''Run the tests'''
    usage = 'Correct usage: \n'
    usage += '    python code.py <input-image> <method>\n';
    usage += 'Methods available: \n'
    usage += '    1 - Blob detection \n'
    usage += '    2 - Harris corner detection \n'


    # Check arguments (first argument is invoked name of program)
    if len(argv) != 3:
        print usage;
        return 1;

    imagepath = argv[1];
    methodno = argv[2];
    methodno = int(methodno);

    # Run the programes with sigmas
    if(methodno == 1):
        blobDetection(imagepath);
    elif(methodno == 2):
        harrisCorners(imagepath);
    else:
        print 'Incorrect method number called';

    # show plot
    plt.show();
  
    # Return 0 to indicate normal termination
    return 0;

def blobDetection(imagepath):

    print('Blob detection called');
    # Read the image from file
    im = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE);

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create();
     
    # Detect blobs.
    keypoints = detector.detect(im);
     
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS);
     
    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)

    return 0;

def harrisCorners(imagepath):

    print('Harris corner detection called');
    # Read the image from file
    img = cv2.imread(imagepath);

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]

    cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    return 0;

if __name__ == '__main__':    
    sys.exit(main(sys.argv))
else:
    print __name__
