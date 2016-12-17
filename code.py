#!/usr/bin/python2
# -*- coding: utf-8 -*-

import sys, os, cv2
import scipy as sp
import scipy.misc as spmisc
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt


def getWindow(width, point, image):
    '''Returns the patch centers around point'''
    row, col = point

    row = int(row)
    col = int(col)

    result = []

    for i in range(width):
        for j in range(width):
            x = int(row - width/2 + i)
            y = int(col - width/2 + j)

            result.append(image[x][y])
            
    return result

def main(argv):
    '''Run the tests'''
    usage = 'Correct usage: \n'
    usage += '    python code.py <method>\n';
    usage += 'Methods available: \n'
    usage += '    1 - Blob detection \n'
    usage += '    2 - Harris corner detection \n'
    usage += '    3 - Matching\n'


    # Check arguments (first argument is invoked name of program)
    if len(argv) != 2:
        print usage;
        return 1;

    imagepaths = ["./Img001_diffuse_smallgray.png","./Img002_diffuse_smallgray.png","./Img009_diffuse_smallgray.png"];
    methodno = argv[1];
    methodno = int(methodno);

    # Run the programes
    if(methodno == 1):
        blobDetection(imagepaths);
    elif(methodno == 2):
        harrisCorners(imagepaths);
    elif(methodno == 3):
        matching(imagepaths);
    else:
        print 'Incorrect method number called';
  
    # Return 0 to indicate normal termination
    return 0;

def blobDetection(imagepaths):

    print('Blob detection called');
    i=1;
    for imagepath in imagepaths:
        # Read the image from file
        im = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE);

        # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector_create();
         
        # Detect blobs.
        keypoints = detector.detect(im);
         
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS);
         
        # Show keypoints side by side (uncomment 2 lines and the last plt.show after for loop)
        # plt.subplot(1,3,i),plt.imshow(im_with_keypoints);
        # plt.title(imagepath), plt.xticks([]), plt.yticks([]);
        i+=1;

        # Show only one image at a time:
        plt.imshow(im_with_keypoints);
        plt.title('Blob detection - '+ imagepath), plt.xticks([]), plt.yticks([]);
        plt.show();
    # plt.show();

    return 0;

def harrisCorners(imagepaths):

    print('Harris corner detection called');
    i=1;
    for imagepath in imagepaths:
        # Read the image from file
        img = cv2.imread(imagepath);

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);

        gray = np.float32(gray);
        dst = cv2.cornerHarris(gray,2,3,0.04);

        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None);

        # Threshold for an optimal value, it may vary depending on the image.
        threshold = 0.003;
        img[dst>threshold*dst.max()]=[255,0,0]

        # Show keypoints side by side (uncomment 2 lines and the last plt.show after for loop)
        # plt.subplot(1,3,i),plt.imshow(img);
        # plt.title(imagepath), plt.xticks([]), plt.yticks([]);
        i+=1;

        # Show only one image at a time:
        plt.imshow(img);
        plt.title('Corner detection - '+ imagepath+', Threshold: ' + repr(threshold)), plt.xticks([]), plt.yticks([]);
        plt.show();
    # plt.show();

    return 0;

def matching(imagepaths, N = 45, k = 2):

    print('Matching called')

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create()

    # Set up the matcher
    # Is this the correct distance measurement?
    matcher = cv2.BFMatcher(cv2.NORM_L2SQR, crossCheck=False)

    def extractDescriptors(imagepath):
        img = cv2.imread(imagepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect blobs.
        keypoints = detector.detect(gray)
        
        descriptors = []

        for kp in keypoints:
            try:
                # Will throw if the patch extends outside of the image
                # In this case, discard the descriptors
                # Should we do something else?
                w = getWindow(N, kp.pt, gray)
                descriptors.append(w)
            except:
                pass

        descriptors = np.array(descriptors, dtype=np.float32)

        return (gray, keypoints, descriptors)

    # The first image is the reference one
    referenceImage, referenceKeypoints, referenceDescriptors = extractDescriptors(imagepaths[0])

    for imagepath in imagepaths[1:]:
        image, keypoints, descriptors = extractDescriptors(imagepath)

        # Compute the matches between the descriptors
        matches = matcher.knnMatch(descriptors, referenceDescriptors, k)

        goodMatches = []

        for descriptorMatches in matches:
            # Sort them in the order of their distance.
            sortedDescriptorMatches = sorted(descriptorMatches, key = lambda x:x.distance)

            # TODO
            # To select a correspondence between a point in image A and points from
            # image B, you should pick the point pairs with the smallest dissimilarity measure.
            # However, you may choose not to accept this if:
            #     The ratio of dissimilarity between the best and the second best match is
            #     too close to 1 (say above 0.7).
            #     2. If the best B-image match y to an A-image point x has a best A-image
            #     match z such that x 6= z (thus left-to-right matching and right-to-left
            #     matching must agree).

            # Save the best match
            goodMatches.append(sortedDescriptorMatches[0])

        image = cv2.drawMatches(image, keypoints, referenceImage, referenceKeypoints, goodMatches, np.array([]), flags=2)

        plt.title(imagepaths[0] + ' - ' + imagepath),plt.imshow(image),plt.show()


if __name__ == '__main__':    
    sys.exit(main(sys.argv))
else:
    print __name__
