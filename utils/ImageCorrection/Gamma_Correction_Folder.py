import os
import cv2


from skimage import io


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, img_as_float
from skimage import exposure

#this program deletes the Exif data



#save in test folder 
cv_img = []
cv_img_2 = []
img_name = []

#input path 
pathname1= './outputs_contrasting/'

#output path 
output_path = './output_contrasting_gamma/'

count = 0
for_loop_count = 0 
for item1 in sorted(os.listdir(pathname1)):    
            if item1.endswith('.jpg'):
            
                path1 = os.path.join(pathname1,item1)

                img = io.imread(path1,plugin='matplotlib')
                #path2 = os.path.join(pathname2,item1)
                #read image 1 and 2
                count = count +1

                #read image 1 and 2
                #image1 = cv2.imread(path1)
                cv_img.append(img)
                img_name.append(item1)
                #print(item1)
                #print(image1)
                
                #image2 = cv2.imread(path2)
                #cv_img_2.append(image2)
                #img_name.append((item1))

#remove spaces
for i in range(0,len(img_name)):
    img_name[i] = img_name[i].replace(" ", "_")
    

for i in range(0,len(cv_img)):
    gamma_corrected = exposure.adjust_gamma(cv_img[i], 2)
    filename = os.path.join(output_path, img_name[i])
    io.imsave(filename, gamma_corrected)
