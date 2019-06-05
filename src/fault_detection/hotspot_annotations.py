#Import necessary modules 
import cv2
import glob
import numpy as np
import os
import json
from imutils import contours
from skimage import measure
import imutils


dict_dir = {}
dict_1 = {}

def hotspot(path):
    #Module detects hotspots from image. Hotspots correspond to Cell Single Fault and Cell Multi Fault. 
    #If there is a single hotspot on a panel it is Cell Single Fault. If there are multiple spots on the same panel
    # it is Cell Multi Fault. 
    print(path)
    #pick images from directory and add to a array. Extract filename from directory and add them to a list
    cv_img = []
    img_name = []
    files = [item for sublist in [glob.glob(path + ext) for ext in ["/*.jpg", "/*.JPG", "/*.png", "/*.PNG"]] for item in sublist]
    for img in files:
        n= cv2.imread(img)
        cv_img.append(n)
        img_name.append(os.path.basename(img))

    #Iterate over images
    for j in range(0,len(cv_img)):
        
        dict_img = {}

        points = []
        #Convert image to single channel
        gray = cv2.cvtColor(cv_img[j], cv2.COLOR_BGR2GRAY)
        #Blur image to reduce noise 
        blurred = cv2.GaussianBlur(gray, (3,3), 0)
        #Threshold the image over a set of values
        thresh = cv2.threshold(blurred, 210, 255, cv2.THRESH_BINARY)[1]
        
        #From the thresholded image pick only blobs which have specified neighbors.
        labels = measure.label(thresh, neighbors=4, background=0)
        
        #Create a black mask where the blobs obatined in labels can be pasted
        mask = np.zeros(thresh.shape, dtype="uint8")

        #Filename extract without extension 
        fname=os.path.splitext(img_name[j])[0]
        
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

        #Contours are applied and the Circle is drawn on the image from left to right.
        for (i, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            #assumption that hostpots are not bigger than 10. If bigger they will be classified as Hotpatch. 
            if (radius > 1  and radius <10) :
                cv2.circle(cv_img[j], (int(cX), int(cY)), int(radius+10),(0, 255, 0), 2)
#                cv2.putText(cv_img[j], "#{}".format(i + 1), (x+20, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                points.append((cX,cY,radius))
            
#        #Path name where the image will be stored
        output_dir = path + '/Hotspots/'
        cv2.imwrite(output_dir +img_name[j],cv_img[j])
#        Convert points list float values to integers

        dict_dir[fname] = dict_img
        dict_img['Cell'] = points        
    return dict_dir


        
#Call to function
dict_1 = hotspot('./IR_Images') 


with open('Hotspots.json', 'w') as f:
            
#    json.dump(x,f)
    json.dump(dict_1,f, indent=4, sort_keys = True)
