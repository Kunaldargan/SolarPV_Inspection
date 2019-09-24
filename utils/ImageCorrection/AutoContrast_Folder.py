#!/usr/bin/env python
# coding: utf-8

#sftp://sameer@192.168.0.104/home/sameer/Jupyter_Notebook/data/Faults_on_Stitched_image/Output_stitched



import os
import cv2
from skimage import io
import numpy as np
from skimage import data, img_as_float
from skimage import exposure


#save in test folder 
cv_img = []
img_name = []
count=0

input_path= input("Input directory : ") #'/home/galactica/Sameer_Data/jupyter_notebooks/Eric_Dataset/Exif/B1/'
output_path = input("Output directory : ") #'/home/galactica/Sameer_Data/jupyter_notebooks/Eric_Dataset/ODM/1_Gap/'

for item1 in sorted(os.listdir(input_path)):    
            if item1.endswith('.jpg'):
            
                path = os.path.join(input_path,item1)

                img = io.imread(path)

                #read image 1 and 2
                count = count+1

                #read image 1 and 2
                cv_img.append(img)
                img_name.append(item1)
                


for i in range(0,len(cv_img)):
    gamma_corrected = exposure.adjust_gamma(cv_img[i], 2)
    filename = os.path.join(output_path, img_name[i])
    io.imsave(filename, gamma_corrected)

