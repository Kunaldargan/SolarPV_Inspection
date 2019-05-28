#Import necessary modules 
import cv2
import glob
import numpy as np
import os

from imutils import contours
from skimage import measure
import imutils

#cv2 to handle images, glob for directory, imutils for image processing, skimage measure for counting neighbors, 


def count_panel(path):
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
    
    #Itearate over images list for segmentation and counting. 

    for j in range(0,(len(cv_img))):

        img = cv_img[j].copy()
        rgbimg = cv_img[j].copy()
        stitchimg = cv_img[j].copy()
        #Morph_Open
        kernel_3 = np.ones((11,11),np.uint8)
        img = cv2.morphologyEx(cv_img[j], cv2.MORPH_OPEN, kernel_3)

        #Convert image to single channel and change color colding to HSV
        hsv_img = cv2.cvtColor(cv_img[j], cv2.COLOR_BGR2HSV)

        #Threshold blue color only to segment the panels from HSV image obtained
        COLOR_MIN = np.array([110,50,50],np.uint8)  
        COLOR_MAX = np.array([130,255,255],np.uint8)
        frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
        
        #apply mask to the binary image o view panels.
        mask = frame_threshed
        res = cv2.bitwise_and(cv_img[j],cv_img[j], mask= mask) 

        #Morph_Erode 
        kernel = np.ones((7,7),np.uint8)
        erosion = cv2.erode(res,kernel,iterations = 1)
        
        #Morph_Dilate 
        kernel2 = np.ones((31,31),np.uint8)
        dilation = cv2.dilate(erosion,kernel2,iterations = 1)

        #Convert dilated image to single channel to threshold
        gray = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)

        ret,thresh2 = cv2.threshold(gray,0,127,cv2.THRESH_OTSU)

        #threshold the image and obtain the binary image
        (thresh3, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        #Morph_Erdoe
        kernel_2 = np.ones((5,5),np.uint8)
        erosion_2 = cv2.erode(im_bw,kernel_2,iterations = 2)

        #Morph_Dilate
        kernel = np.ones((7,7),np.uint8)
        dilation = cv2.dilate(rgbimg,kernel,iterations = 1)

        #Impose the binary mask on the original image
        gray2 = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
        new = erosion_2 * gray2
        #Threshold the image 
        ret, thresh3 = cv2.threshold(new,
                        127, 255, cv2.THRESH_BINARY)

        #Contours application to the image containing panels only. 
        contours2,hierachy,=cv2.findContours(thresh3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #Initialize the count variable to count the number of panels. 
        count=0
        for c in contours2:
            #Complete boxes to the contours which have created incomplete boxes.
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.00001 * peri, True)
            #Add a bounding rectangle to the contours
#            x,y,w,h = cv2.boundingRect(approx)
#            #Use the height and width of the bounding box to remove unwanted boxes in the background
#            if (20 < w < 90 and 20 < h < 90):
#                cv2.rectangle(stitchimg,(x, y), (x+w, y+h), (0, 255, 0), 2)
#                count+=1
            #Using minAreaRect to solve rotated rectangle issue
            rect = cv2.minAreaRect(approx)
            w, h = rect[1]

            if (15 < w < 90 and 15 < h < 90):
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(stitchimg,[box],0,(0,255,0),2)
                count+=1
        print(count)
        output_dir = path + '/Output/'
        #Write the image in the path's output folder
        cv2.imwrite(output_dir +img_name[j],stitchimg)


#Call to path function 
        
count_panel('/home/sameer/Galactica_Solar/Solar_Inspection/Images/RGB')
