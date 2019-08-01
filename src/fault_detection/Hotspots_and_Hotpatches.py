import os
import cv2 

import cv2
import glob
import numpy as np
import os
import json
from imutils import contours
from skimage import measure
import imutils


#pick files from folders 
#save in test folder 
cv_img = []
cv_img_2 = []
img_name = []


pathname1= '/home/sameer/Galactica_Solar/Lapeer_Faults/Combined'
pathname2 = '/home/sameer/Galactica_Solar/Lapeer_Faults/Images'

matches = 0 
count = 0
for_loop_count = 0 
for item1 in sorted(os.listdir(pathname1)):    
            if item1.endswith('.jpg'):
            
                path1 = os.path.join(pathname1,item1)

                path2 = os.path.join(pathname2,item1)
                #read image 1 and 2
                count = count +1

                #read image 1 and 2
                image1 = cv2.imread(path1)
                cv_img.append(image1)
                img_name.append(item1)
                #print(item1)
                #print(image1)
                
                image2 = cv2.imread(path2)
                cv_img_2.append(image2)
                #img_name.append((item1))


#assign output variable 

output_path_h = '/home/sameer/Galactica_Solar/Lapeer_Faults/Output/'

for j in range(0,len(cv_img)):
    #Convert image to sngle channel image
    #insert hotspots here 
    points = []
    #Convert image to single channel
    gray = cv2.cvtColor(cv_img[j], cv2.COLOR_BGR2GRAY)
    #Blur image to reduce noise 
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    #Threshold the image over a set of values
    ######Change parameters here 
    thresh = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)[1]
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
    print(img_name[j])
    #Contours are applied and the Circle is drawn on the image from left to right.
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        #print(cv2.minEnclosingCircle(c))
        #assumption that hostpots are not bigger than 10. If bigger they will be classified as Hotpatch. 
        if (radius > 2 and radius<10) :
            cv2.circle(cv_img_2[j], (int(cX), int(cY)), int(radius+5),(0, 255, 0), 2)
            #cv2.circle(cv_img[j], (int(cX), int(cY)), int(radius+5),(0, 255, 0), 2)
#                cv2.putText(cv_img[j], "#{}".format(i + 1), (x+20, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            points.append((cX,cY,radius))
    
    
    #hotpatches

    gray = cv2.cvtColor(cv_img[j], cv2.COLOR_BGR2GRAY)

    #Blur the image to reduce noise
    blurred = cv2.GaussianBlur(gray, (3,3), 0)

    #Threshold the image over a particular range of pixel values
    #######Change parameters here 
    thresh = cv2.threshold(blurred, 245, 255, cv2.THRESH_BINARY)[1]

    #Morph_Dilate to bring out the pixels
    thresh = cv2.dilate(thresh, None, iterations=4)

    #From the thresholded image pick only blobs which have specified neighbors.
    labels = measure.label(thresh, neighbors=4, background=0)

    #Create a black mask where the blobs obatined in labels can be pasted
    mask = np.zeros(thresh.shape, dtype="uint8")

    fname=os.path.splitext(img_name[j])[0]

    for label in np.unique(labels):
        if label == 0:
            continue

        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        #Paste the blob only if the value of pixels in it is above a certain threshold value
        if numPixels > 300:
            mask = cv2.add(mask, labelMask)
    #Apply contours on the mask image which contains the blobs
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #Contours are applied and the Convex hull is drawn on the image from left to right.
    for (i, c) in enumerate(cnts):
        area = cv2.contourArea(c)
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        hull = cv2.convexHull(c)
        #Draw convex hull on the blobs of the mask image depending on the area it is a diode fault or string fault
        if (area<1000):#assumption

            cv2.drawContours(cv_img_2[j], [hull], -1, (0, 255, 0), 3)                    
            cv2.putText(cv_img_2[j], "Diode Fault", (x+20, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
            b=hull.tolist()
            #diode_points.append(b)
        elif(area<100000):
            cv2.drawContours(cv_img_2[j], [hull], -1, (0, 255, 0), 3)              
            cv2.putText(cv_img_2[j], "String Fault", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
            c=hull.tolist()
            #string_points.append(c)
            
    output_file = os.path.join(output_path_h,img_name[j])
    cv2.imwrite(output_path_h  + img_name[j],cv_img_2[j])
            
