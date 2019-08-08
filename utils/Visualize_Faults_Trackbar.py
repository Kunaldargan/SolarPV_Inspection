from __future__ import print_function
from __future__ import division
import cv2

#Extra imports 
import glob
import numpy as np
import os
import json
from imutils import contours
from skimage import measure
import imutils


#import argparse
alpha_slider_max = 255
title_window = 'Hotspots'
def on_trackbar(val):
    #alpha = val / alpha_slider_max
    #beta = ( 1.0 - alpha )
    #dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
    #cv2.imshow(title_window, dst)

    #insert hotspots here 
        #Convert image to sngle channel image
    #dict_img = {}
    #insert hotspots here 
    #points = []
    #Convert image to single channel
    src1 = cv2.imread('150.jpg')
    gray = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
    #Blur image to reduce noise 
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    #Threshold the image over a set of values
    thresh = cv2.threshold(blurred, val, 255, cv2.THRESH_BINARY)[1]
    #cv2.imwrite(output_dir +'1'+img_name[j],thresh)
    #From the thresholded image pick only blobs which have specified neighbors.
    labels = measure.label(thresh, neighbors=4, background=0)

    #Create a black mask where the blobs obatined in labels can be pasted
    mask = np.zeros(thresh.shape, dtype="uint8")

    #Filename extract without extension 
    #fname=os.path.splitext(img_name[j])[0]

    for label in np.unique(labels):

        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        #Paste the blob only if the value of pixels in it is above a certain threshold value
        if numPixels > 300:#Assumption
            mask = cv2.add(mask, labelMask)
    #Apply contours on the mask image which contains the blobs
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #print(img_name[j])
    #Contours are applied and the Circle is drawn on the image from left to right.
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        #print(cv2.minEnclosingCircle(c))
        #assumption that hostpots are not bigger than 10. If bigger they will be classified as Hotpatch. 
        if (radius > 2 and radius<10) :
            cv2.circle(src1, (int(cX), int(cY)), int(radius+5),(0, 255, 0), 2)
            #cv2.circle(cv_img[j], (int(cX), int(cY)), int(radius+5),(0, 255, 0), 2)
            #cv2.putText(cv_img[j], "#{}".format(i + 1), (x+20, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            #points.append((cX,cY,radius))
    cv2.imshow(title_window, src1)





#src1 = cv2.imread('8.jpg')
#src2 = cv2.imread('22.jpg')

cv2.namedWindow(title_window)
trackbar_name = 'Threshold Value %d' % alpha_slider_max
cv2.createTrackbar(trackbar_name, title_window , 0, alpha_slider_max, on_trackbar)
# Show some stuff
on_trackbar(0)
# Wait until user press some key
cv2.waitKey()
#k = cv2.waitKey(20) & 0xFF
#if k == 27: break
#elif k == ord('a'):
