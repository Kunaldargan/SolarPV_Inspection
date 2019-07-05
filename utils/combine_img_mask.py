#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:41:50 2019

@author: sameer
"""


#extract single channel only easier for plotting
import cv2
import glob
import numpy as np
import os
cv_img_1 = []
cv_img_2 = []

#
#for img in glob.glob("./Images/*.jpg"):
#    n= cv2.imread(img,0)
#    cv_img_1.append(n)
#
#for img in glob.glob("./Masks/*.jpg"):
#    n= cv2.imread(img,0)
#    cv_img_2.append(n)
#
#
#for i in range(0,len(cv_img_1)):
#    
#    mask = cv_img_2[i]
#    res = cv2.bitwise_and(cv_img_1[i],cv_img_1[i], mask= mask) 
#    cv2.imwrite(str(i) + '.jpg',res)
#    
#
#for img in glob.glob("./Images/*.jpg"):
#for img in glob.glob("./Images/*.jpg"):
#
#os.listdir(path)


path = './Images'
list1 = []
file_names = []

list1 = os.listdir(path)
list2 = []
for fname in list1:
    path2 = os.path.join(path,fname)
    file_names.append(fname)
    list2.append(path2)
sorted(list2)

for img in list2:
    n= cv2.imread(img,0)
    cv_img_1.append(n)


    
path = './Masks'
list1 = []
list1 = os.listdir(path)
list3 = []
for fname in list1:
    path2 = os.path.join(path,fname)
    list3.append(path2)
sorted(list3)

for img in list3:
    n= cv2.imread(img,0)
    cv_img_2.append(n)




for i in range(0,len(cv_img_1)):
    
    mask = cv_img_2[i]
    res = cv2.bitwise_and(cv_img_1[i],cv_img_1[i], mask= mask) 
    cv2.imwrite(str(i) + '.jpg',res)
    











#
#
#img1 = cv2.imread('2.jpg', 0)
#img2 = cv2.imread('2m.jpg',cv2.IMREAD_UNCHANGED)
#
#new = img2 * img1 
#
#
#
#mask = img2
#res = cv2.bitwise_and(img1,img1, mask= mask) 
#cv2.imwrite('mask.jpg',res)