#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:34:09 2019

@author: sameer
"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Import necessary modules 
import cv2
import glob
import numpy as np
import os
import json

from imutils import contours
from skimage import measure
import imutils

#cv2 to handle images, glob for directory, imutils for image processing, skimage measure for counting neighbors, 
dict_dir = {}


def count_panel(path):
    rotation = 0
    print(path)
    #pick images from directory and add to a array. Extract filename from directory and add them to a list
    cv_img = []
    img_name = []
    files = [item for sublist in [glob.glob(path + ext) for ext in ["/*.jpg", "/*.JPG", "/*.png", "/*.PNG"]] for item in sublist]
    for img in files:
        n= cv2.imread(img)
        cv_img.append(n)
        img_name.append(os.path.basename(img))
        print(os.path.basename(img))
    
    #Iterate over images list for segmentation and counting. 

    for j in range(0,len(cv_img)):

        points =[]
        dict_img = {}
        img = cv_img[j].copy()
        rgbimg = cv_img[j].copy()
        stitchimg = cv_img[j].copy()
        #Morph_Open
        kernel_opening = np.ones((17,17),np.uint8)
        img = cv2.morphologyEx(cv_img[j], cv2.MORPH_OPEN, kernel_opening)
#        cv2.imwrite('1_Opening.jpg',img)

        #Convert image to single channel and change color colding to HSV
        hsv_img = cv2.cvtColor(cv_img[j], cv2.COLOR_BGR2HSV)

        #Threshold blue color only to segment the panels from HSV image obtained
        COLOR_MIN = np.array([110,50,50],np.uint8)  
        COLOR_MAX = np.array([130,255,255],np.uint8)
        frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
#        cv2.imwrite('2_frame_threshed.jpg',frame_threshed)
        
        #apply mask to the binary image o view panels.
        mask = frame_threshed
        res = cv2.bitwise_and(cv_img[j],cv_img[j], mask= mask) 
#        cv2.imwrite('3_res.jpg',res)
        #Morph_Erode 
        kernel_erode = np.ones((3,3),np.uint8)
        erosion = cv2.erode(res,kernel_erode,iterations = 1)
#        cv2.imwrite('4_erosion.jpg',erosion)
        #Morph_Dilate 
        kernel_dilate = np.ones((31,31),np.uint8)
        dilation = cv2.dilate(erosion,kernel_dilate,iterations = 1)
#        cv2.imwrite('5_dilation.jpg',dilation)
        #Convert dilated image to single channel to threshold
        gray = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)

        ret,thresh2 = cv2.threshold(gray,0,127,cv2.THRESH_OTSU)

        #threshold the image and obtain the binary image
        (thresh3, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#        cv2.imwrite('6_im_bw.jpg',im_bw)
        #Morph_Erdoe
        kernel_erode_2 = np.ones((5,5),np.uint8)
        erosion_2 = cv2.erode(im_bw,kernel_erode_2,iterations = 2)
#        cv2.imwrite('7_erosion_2.jpg',erosion_2)
        #Morph_Dilate
        kernel_dilate_2 = np.ones((7,7),np.uint8)
        dilation = cv2.dilate(rgbimg,kernel_dilate_2,iterations = 1)
#        cv2.imwrite('8_dilation_2.jpg',dilation)
        
        #Impose the binary mask on the original image
        gray2 = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
        new = erosion_2 * gray2
#        cv2.imwrite('9_new.jpg',new)
        #apply canny to new
        edges = cv2.Canny(new,100,250)
#        cv2.imwrite('canny.jpg',edges)
        
        kernel_closing_5 = np.ones((5,5),np.uint8)

        morph_closing = cv2.morphologyEx(new, cv2.MORPH_CLOSE, kernel_closing_5)
#        cv2.imwrite('Morph_CLose.jpg',morph_closing)
        
        #Grdaient try 
        kernel_gradient = np.ones((3,3),np.uint8)
        gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel_gradient)
#        cv2.imwrite('gradient.jpg',gradient)

        #Threshold the image 
        ret, thresh3 = cv2.threshold(new,
                        127, 255, cv2.THRESH_BINARY)
#        cv2.imwrite('10_thresh3.jpg',thresh3)
        #Contours application to the image containing panels only. 
        contours2,hierachy,=cv2.findContours(gradient,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #Initialize the count variable to count the number of panels. 
        fname=os.path.splitext(img_name[j])[0]
        count=0
        rotation=1
        for c in contours2:
            #Complete boxes to the contours which have created incomplete boxes.
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.00001 * peri, True)
            #Add a bounding rectangle to the contours
            if(rotation==0):
                x,y,w,h = cv2.boundingRect(approx)
            #Use the height and width of the bounding box to remove unwanted boxes in the background
                if (85 < w < 500 and  85< h < 500):
                    cv2.rectangle(stitchimg,(x, y), (x+w, y+h), (0, 255, 0), 2)
                    count+=1
                points.append((x,y,x+w,y+h))
            else:
                rect = cv2.minAreaRect(approx)
                w = rect[1][0]
                h = rect[1][1]
                if (50 < w < 500 and 50 < h < 500):
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(stitchimg,[box],0,(0,255,0),3)
                    count+=1
                
        output_dir = path + '/Output/'
        #Write the image in the path's output folder
        cv2.imwrite(output_dir+img_name[j],cv_img[j])
        #points_img contains img of cuurent img. count.

        dict_img['Count'] = count
        dict_img['Points'] = points
        dict_dir[fname] = dict_img
        
        
        
        
    return dict_dir
       
dict1 = {}        
dict_1 =count_panel('./input')



with open('Exif2.json', 'w') as f:
            
#    json.dump(x,f)
    json.dump(dict_1,f, indent=4, sort_keys = True)



