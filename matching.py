#!/ usr /bin /python
# -*- coding: utf-8 -*-
#
# Author : Sriram Ka r thik Badam
# Date : Sep 26 , 2012
#
import sys , os
import cv2
import cv

import numpy as np
#image v a r i a b l e s
image1 = 0
image2 = 0
#c o n s t a n t s
WINDOW_SIZE = 5
TOTAL_FEATURES = 200
#Window size and Th re sh old s f o r s t e r e o.jp g
"""
SSD_WINDOW_SIZE = 41
NCC_WINDOW_SIZE = 53
SSD_THRESHOLD = 1100
SSD_RATIO_THRESHOLD = 0.8
NCC_THRESHOLD = 0.8
NCC RATIO_THRESHOLD = 1.2
"""
#Window size and Th re sh old s f o r b u i l d i n g.jp g
"""
SSD_WINDOW_SIZE = 41
NCC_WINDOW_SIZE = 15
SSD_THRESHOLD = 1300
SSD_RATIO_THRESHOLD = 0.8 5
NCC_THRESHOLD = 0.7 5
NCC RATIO_THRESHOLD = 1.2
"""
#Window size and Th re sh old s f o r sample.jp g
SSD_WINDOW_SIZE = 41
NCC_WINDOW_SIZE = 53
SSD_THRESHOLD = 1100
SSD_RATIO_THRESHOLD = 0.75
NCC_THRESHOLD = 0.78
NCC_RATIO_THRESHOLD = 1.1
#Window size and Th re sh old s f o r tower.jp g
"""
SSD_WINDOW_SIZE = 41
NCC_WINDOW_SIZE = 53
SSD_THRESHOLD = 1300
SSD_RATIO_THRESHOLD = 0.7
NCC_THRESHOLD = 0.7 6
NCC_RATIO_THRESHOLD = 1.0 7
"""
#i n p u t al g o ri t hm = 0 => SSD ; al g o ri t hm = 1 => NCC
algorithm = 0
size = 0#i n p u t al g o ri t hm window size
#f e a t u r e cl a s s
class Feature ( object ) :
    #i ni ti ali zi n g
    def __init__ ( self , x, y):
        self.x = x
        self.y = y
        self.match = -1
        self.score = -1
        self.neighbor = 0
"""
Corner Detector function calculates the Corner Responses for
each p i x e l o f the image.By a p pl yi n g non-maximum s u p p r e s si o n
we ge t the i n t e r e s t p oi n t s
"""
def harrisCornerDetector (image ) :
    image_height = np.shape(image)[0]
    image_width = np.shape(image)[1]
    #image d e r i v a t i v e al o n g x
#    image_der_x = cv.CreateImage ( ( image_width , image_height ) , cv.IPL_DEPTH_32F , 1)
    #image d e r i v a t i v e al o n g y
#    image_der_y = cv.CreateImage ( ( image_width , image_height ) , cv.IPL_DEPTH_32F , 1)
    #g re y s c a l e
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #A p pli e s S o bel Operator t o g e t the image d e r i v a t i v e s al on g x , y
#    cv.Sobel ( grayscale_image1 , image_der_x , 1 ,0, 3 )
    image_der_x = cv2.Sobel(grayscale,cv2.CV_32F,1,0,3)
#    cv.Sobel ( grayscale_image1 , image_der_y , 0, 1, 3 )
    image_der_y = cv2.Sobel(grayscale,cv2.CV_32F,1,0,3)
    #C al c ul a t e s C
    global WINDOW_SIZE #size o f mask
    C = cv.CreateMat ( 2 , 2 , cv.CV_64FC1 )
    #v a r i a b l e s f o r SVD
    U = cv.CreateMat ( 2 , 2 , cv.CV_64FC1 )
    D = cv.CreateMat ( 2 , 2 , cv.CV_64FC1 )
    I_x=0
    I_xy = 0
    I_y=0
    R = cv.CreateMat ( image_width , image_height , cv.CV_64FC1 )
    cv.Zero (R)
    #C o n s t r u c t s the v a ri a n c e Matrix a t each p i x e l
    #l o o p s through each p i x e l
    print "Calculating the Covariance"
    for i  in range ( int(WINDOW_SIZE/ 2 ) , image_width - int(WINDOW_SIZE/2 + 1 ) ) :
        for j in range ( int(WINDOW_SIZE/ 2 ) , image_height - int(WINDOW_SIZE/2 + 1 ) ) :
            I_x=0
            I_xy = 0
            I_y=0
            for index_x in range(-int(WINDOW_SIZE/ 2 ) , int(WINDOW_SIZE/2 + 1 ) ) :
                for index_y in range(-int(WINDOW_SIZE/ 2 ) , int(WINDOW_SIZE/2 + 1 ) ) :
                    I_x=I_x + image_der_x[ j + index_y , i + index_x ] * image_der_x[ j + index_y, i + index_x ] / WINDOW_SIZE
                    I_xy = I_xy + image_der_x[ j + index_y , i + index_x ] * image_der_y[ j + index_y, i + index_x ] / WINDOW_SIZE
                    I_y=I_y + image_der_y[ j + index_y , i + index_x ] * image_der_y[ j + index_y, i + index_x ] / WINDOW_SIZE
                    cv.mSet(C, 0 , 0 , I_x )
                    cv.mSet (C, 1 , 0 , I_xy )
                    cv.mSet (C, 0 , 1 , I_xy )
                    cv.mSet (C, 1 , 1 , I_y )
                    #svd
                    cv.SVD(C, D, U, None , 0 )
                    lambda1 = cv.mGet(D, 0 , 0 )
                    lambda2 = cv.mGet(D, 1 , 1 )
                    #corner Response
                    cv.mSet (R, i , j , lambda1*lambda2 - 0.04* np.power ( ( lambda1+lambda2 ) , 2 ) )
    R_final = cv.CreateMat ( image_width , image_height , cv.CV_64FC1 )
    cv.Copy (R, R_final )
    #a p pl y s non maximum s u p p r e s si o n -> el em e n t s o t h e r than maximum v al u e s i n a window are removed
    for i  in range ( int(WINDOW_SIZE/ 2 ) , image_width-int(WINDOW_SIZE/ 2 ) ) :
        for j in range ( int(WINDOW_SIZE/ 2 ) , image_height-int(WINDOW_SIZE/ 2 ) ) :
            for index_x in range(-int(WINDOW_SIZE/ 2 ) , int(WINDOW_SIZE/2 + 1 ) ) :
                for index_y in range(-int(WINDOW_SIZE/ 2 ) , int(WINDOW_SIZE/2 + 1 ) ) :
                    R_value = cv.mGet (R, i , j )
                    R_neighbor_value = cv.mGet(R, i + index_x , j + index_y )
                    if R_value != 0 and R_value < R_neighbor_value :
                        cv.mSet ( R_final , i , j , 0)
                        break
    return R_final
"""
Creates the feature descriptor for each interest point
"""
def getFeatures (R, image, features , threshold ) :
    global algorithm
    global TOTAL_FEATURES
    global size
    if algorithm == 0 :
        size = SSD_WINDOW_SIZE
    if algorithm == 1 :
        size = NCC_WINDOW_SIZE
    feature_number = 0
    for i in range ( int ( size /2) , np.shape(image)[1]-int ( size /2) ) :
        for j in range ( int ( size /2) , np.shape(image)[0]-int ( size /2) ) :
            if cv.mGet(R, i , j ) > threshold : #threshold on the c o rne r re sp on se to prune interest points
                if feature_number < TOTAL_FEATURES:
                    feature_number = feature_number + 1
                    features.append ( Feature( i , j ) ) #adds the f e a t u r e t o the f e a t u r e l i s t
    print "number of features: " , feature_number
"""
f i l l s the ’ neighbor ’ variable with the neighboring pixel values of a given pixel.
"""
def getNeighbor (x, y, grayscaleimage , neighbor ) :
    global size
    for i in range(-int ( size /2) , int ( size /2+1) ) :
        for j in range(-int ( size /2) , int ( size /2+1) ) :
            cv.mSet ( neighbor , int ( size /2) + j , int ( size /2) + i , grayscaleimage[ y+j , x+i ] )
"""
Computes the SSD S c o re
"""
def ssdScore ( f1 , f2 ) :
    global size #size o f SSD window
    #subtracts f2 from f1
    sub_f1_f2 = cv.CreateMat ( size , size , cv.CV_64FC1 )
    cv.Sub ( f1 , f2 , sub_f1_f2 )
    
    #square and add
    f1_f2_square = cv.CreateMat ( size , size , cv.CV_64FC1 )
    cv.Pow( sub_f1_f2 , f1_f2_square , 2)
    score = cv.Sum(f1_f2_square)
    return score[0]/( size * size )
#"""
#Computes the NCC S c o re
#"""
#def nccScore ( f1 , f2 ) :
#global size #size o f NCC window
#mean1 = cv.Avg ( f 1 )
#mean2 = cv.Avg ( f 2 )
#f1_sub_mean = cv.CreateMat ( size , size , cv.CV_64FC1 )
#f2_sub_mean = cv.CreateMat ( size , size , cv.CV_64FC1 )
#cv.SubS ( f1 , mean1 , f1_sub_mean ) ;
#cv.SubS ( f2 , mean2 , f2_sub_mean ) ;
##c al c ul a t e s numerator o f the ncc score f r a c ti o n
#ncc numerator = cv.CreateMat ( size , size , cv.CV_64FC1 )
#cv.Mul ( f1_sub_mean , f2_sub_mean , ncc numerator )
#numerator = cv.Sum( ncc numerator )
##c al c ul a t e s denominator
#16
#magnitude1 = cv.CreateMat ( size , size , cv.CV_64FC1 )
#magnitude2 = cv.CreateMat ( size , size , cv.CV_64FC1 )
#cv.Pow( f1_sub_mean , magnitude1 , 2)
#sum1 = cv.Sum( magnitude1 )
#cv.Pow( f2_sub_mean , magnitude2 , 2)
#sum2 = cv.Sum( magnitude2 )
#denominator = np.s q r t ( sum1[ 0 ] * sum2[ 0 ] )
#score = numerator[0]/ denominator
#return score
"""
prints a matrix
"""
def printMatrix(mat ) :
    temp_array = np.asarray (mat[ : , : ] )
    print temp_array
"""
main function
"""
def main () :
    global image1
    global image2
    #l o a d s image
#    if len ( sys.argv ) > 2 :
#        filename1 = './view0.png'
#        filename2 = './view1.png'
#        image1 = cv.LoadImage ( filename1 , cv.CV_LOAD_IMAGE_UNCHANGED)
#        image2 = cv.LoadImage ( filename2 , cv.CV_LOAD_IMAGE_UNCHANGED)
#    if image1 == 0 :
#        print " enter a valid Image Path for Image 1"
#    if image2 == 0 :
#        print " enter a valid Image Path for Image 2"
#    else :
#        print "Enter two valid image f i l e paths as argument"
    filename1 = './Img001_diffuse_smallgray.png'
    filename2 = './Img002_diffuse_smallgray.png'
    image_1 = cv.LoadImage ( filename1 , cv.CV_LOAD_IMAGE_UNCHANGED)
    
    image_2 = cv.LoadImage ( filename2 , cv.CV_LOAD_IMAGE_UNCHANGED)
    image1 = cv2.imread(filename1)
    image1_height = np.shape(image1)[0]
    image1_width = np.shape(image1)[1]
    image2 = cv2.imread(filename2)
    #a p pli e s corner Detector
    R1 = harrisCornerDetector( image1 )
    R2 = harrisCornerDetector( image2 )
    #a s k s u s e r f o r the al g o ri t hm c h oi c e
    print " type 0 t o s e l e c t SSD and 1 t o s e l e c t NCC"
    global algorithm
    algorithm = int ( raw_input () )
    #C r e a t e s f e a t u r e D e s c ri p t o r s
    features1 =[]
    features2 =[]
    #f i l l s the feature array
    #b uil di n g.jpg
    """
    getFeatures (R1 , image1 , features1 , 7 e9 )
    getFeatures (R2 , image2 , features2 , 3 e10 )
    """
    #sample.jp g
    getFeatures (R1 , image1 , features1 , 7e9 )
    getFeatures (R2 , image2 , features2 , 10e9 )
    #o t h e r s
    """
    getFeatures (R1 , image1 , features1 , 7 e8 )
    getFeatures (R2 , image2 , features2 , 8 e8 )
    """
    grayscaleimage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayscaleimage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    global size , SSD_THRESHOLD, SSD_RATIO_THRESHOLD
    global NCC_THRESHOLD, NCC_RATIO_THRESHOLD
    #Computes neighbors of each interest point and puts them in the f e a t u r e d e s c r i p t o r
    global TOTAL_FEATURES
    for i  in range (TOTAL_FEATURES) :
        f1 = cv.CreateMat ( size , size , cv.CV_64FC1 )
        f2 = cv.CreateMat ( size , size , cv.CV_64FC1 )
        if i < len ( features1 ):
            x1 = features1[ i ].x
            y1 = features1[ i ].y
            getNeighbor ( x1 , y1 , grayscaleimage1 , f1 )
            features1[ i ].neighbor = f1
        if i < len ( features2 ):
            x2 = features2[ i ].x
            y2 = features2[ i ].y
            getNeighbor ( x2 , y2 , grayscaleimage2 , f2 )
            features2[ i ].neighbor = f2
    #Finds C o r re sp ondence s
    #if SSD Chosen
    if algorithm == 0 :
        for i in range (len ( features1 ) ) :
            score = 0
            score_min = 1e10
            score_second_min = 1e10
            for j in range (len ( features2 ) ) :
                score = ssdScore ( features1[ i ].neighbor , features2[ j ].neighbor )
                print score
                #upd a te s min , sec ond min
                if score_min > score and score < SSD_THRESHOLD:#SSD_THRESHOLD i s a threshold on s s dscore values
                    score_second_min = score_min
                    score_min = score
                    features1[ i ].score = score_min
                    features1[ i ].match = j
                    features2[ j ].score = score_min
                    features2[ j ].match = i
                #upd a te s sec ond min
                elif score < score_second_min and score > score_min :
                    score_second_min = score
                if score_second_min > 0 and score_min/score_second_min > SSD_RATIO_THRESHOLD: #Threshold on the r a ti o
                    features1[ i ].score = -1
                    features1[ i ].match = -1
                    features2[ j ].score = -1
                    features2[ j ].match = -1
    #if NCC_ch o sen
#    if algorithm == 1 :
#    for i in range (len ( features1 ) ) :
#    score = 0
#    score_max = -1e10
#    score_second_max = -1e10
#    for j in range (len ( features2 ) ) :
#    score = nccScore ( features1[ i ].neighbor , features2[ j ].neighbor )
#    #upd a te s max, second max
#    if score > score_max and score > NCC_THRESHOLD:#NCC_THESHOLD i s a threshold on the NCC s c o r e Values
#    score_second_max = score_max
#    score_max = score
#    features1[ i ].score = score_max
#    features1[ i ].match = j
#    features2[ j ].score = score_max
#    features2[ j ].match = i
#    #upd a te s second max
#    elif score > score_second_max and score < score_max :
#    score_second_max = score
#    if score_second_max > 0 and score_max/score_second_max < NCC RATIO_THRESHOLD:#
#    threshold on r a ti o
#    features1[ i ].score = -1
#    features1[ i ].match = -1
#    features2[ j ].score = -1
#    features2[ j ].match = -1
    #r e c o n s t r u c t the f i n a l image
    print np.shape(image1)[2]
    final_image = cv.CreateImage ((2 * image1_width , image1_height ) , 3 , 4)
    cv.SetImageROI ( final_image , (0 , 0 , image1_width , image1_height ) )
    print image1
    print final_image
    cv.Copy ( image1 , final_image )
    #marks the i n t e r e s t p oi n t s i n the f i r s t p a r t
    for i in range (len ( features1 ) ) :
        cv.Circle ( final_image , ( features1[ i ].x, features1[ i ].y) , 0, (255, 0, 0) , 4)
        cv.ResetImageROI ( final_image )
        cv.SetImageROI ( final_image , (image1_width , 0 , image1_width , image1_height ) )
        cv.Copy ( image2 , final_image )
    #marks the i n t e r e s t p oi n t s i n the sec ond p a r t o f the image
    for i in range (len ( features2 ) ) :
        cv.Circle ( final_image , ( features2[ i ].x, features2[ i ].y) , 0, (255, 0, 0) , 4)
        cv.ResetImageROI ( final_image )
    #shows c o r r e s p o n d e n c e s by drawing a l i n e between matching I n t e r e s t P oi n t s
    count = 0
    for i in range (len ( features1 ) ) :
        if features1[ i ].match != -1:
            count = count + 1
        #dif f e r e n t c ol o r f o r each li n e to d if f e r e n t i a t e between them.
        cv.Line ( final_image , ( features1[ i ].x, features1[ i ].y) , ( features2[ features1[ i ].match ].x + image1_width , 
            features2[ features1[ i ].match].y) , (255*( count%4) , 255*( ( count+1)%4) , 255*( ( count+2)%4) ) , 1 , cv.CV_AA, 0 )
    print "Number of Correspondences = " , count
    cv.SaveImage (" Result.png " , final_image )#r e s u l t s t o r e d i n R e s ul t.png
if __name__ == "__main__" :
    main()