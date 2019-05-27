
#Import necessary modules 
import cv2
import glob
import numpy as np
import os


from imutils import contours
from skimage import measure
import imutils

def hotpatch(path):
    #Module detects hotpatches  from image. Hotpatch correspond to Diode Single Fault and Diode Multi Fault. 
    #If there is a Diode hotpatch on a panel it is Cell Diode Fault. If there are multiple patches on the same panel
    # it is Diode Multi Fault.
    cv_img = []
    img_name = []
    #pick images from directory and add to a array. Extract filename from directory and add them to a list

    files = [item for sublist in [glob.glob(path + ext) for ext in ["/*.jpg", "/*.JPG", "/*.png", "/*.PNG"]] for item in sublist]
    for img in files:
        n= cv2.imread(img)
        cv_img.append(n)
        img_name.append(os.path.basename(img))
        print(os.path.basename(img))
 

    for j in range(0,len(cv_img)):
        #Convert image to sngle channel image
        gray = cv2.cvtColor(cv_img[j], cv2.COLOR_BGR2GRAY)
        
        #Blur the image to reduce noise
        blurred = cv2.GaussianBlur(gray, (3,3), 0)
        
        #Threshold the image over a particular range of pixel values
        thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
        
        #Morph_Dilate to bring out the pixels
        thresh = cv2.dilate(thresh, None, iterations=4)
        
        #From the thresholded image pick only blobs which have specified neighbors.
        labels = measure.label(thresh, neighbors=4, background=0)
        
        #Create a black mask where the blobs obatined in labels can be pasted
        mask = np.zeros(thresh.shape, dtype="uint8")

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
            (x, y, w, h) = cv2.boundingRect(c)
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            #Draw convex hull on the blobs of the mask image
            hull = cv2.convexHull(c)
            cv2.drawContours(cv_img[j], [hull], -1, (0, 0, 255), 3)
#            cv2.putText(cv_img[j], "#{}".format(i), (x+20, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        #Directory where the image containing Hotpatches will be stored
        output_dir = path + '/Hotpatches/'
        cv2.imwrite(output_dir +img_name[j],cv_img[j])
        
hotpatch('/home/sameer/Galactica_Solar/Solar_Inspection/Images/IR')
