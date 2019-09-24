import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import piexif
from PIL import Image

def get_hist(img):

	# get histogram
	hist = cv2.calcHist([img],[0],None,[256],[0,256])
	hist = hist.astype(int)
	return hist


def normalise(img,perc):
	hist = get_hist(img)
	img = img.astype(np.int16)

	l = 0
	u = 255
	h,w = img.shape
	pixels = h*w*perc/100
	curr = 0
	for i in range(0,256):
		if hist[i] + curr < pixels:
			curr+=hist[i]
		else:
			l = i
			break

	curr = 0
	for i in range(255,-1,-1):
		if hist[i] + curr < pixels:
			curr+=hist[i]
		else:
			u = i
			break

	img = (img - l)/(u-l)
	img = img*255
	img[img<0] = 0
	img[img>255] = 255
	img = img.astype(np.uint8)

	return img

################## CODE STARTS FROM HERE ####################
directory = input("soruce dir : ")
output = input("output dir : ")
perc = input("percentage of pixels: ")
for im in os.listdir(directory):
	
	# STEP 1 -> READ ORIGINAL
	img = cv2.imread(os.path.join(directory,im),0)
	exif_dict = piexif.load(os.path.join(directory,im))
	T_norm = normalise(img.copy(),int(perc))
	OpenCVImageAsPIL = Image.fromarray(T_norm)
	OpenCVImageAsPIL.save(os.path.join(output,im), format='JPEG', exif=exif_dict) 
	#cv2.imwrite(os.path.join(output,im),T_norm)

