import numpy as np
import cv2
import glob
import os
from skimage import measure
import imutils


class Counts:
	"""
	Inumpyut: Images or List of Images

	Output: Json file with containing segmented out solar panels - points, counts and ordered set

	Functions:
		segment() : Returns background segmentation points, count of panels and arrays in dictionary
		order_by() : to get order of solar panels, returns dictionary
		array_count(): to get solar arrays counts, returns dictionary
		panel_count(): to get solar panel counts, returns dictionary
	"""

	def __init__(self):
		pass;

	def count_panel(self, img, panel_type="Drapper"):

		rotation = 0;
		points = []
		dict_img = {}

		if (panel_type == "Drapper"):

			#Impose the binary mask on the original image
			gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			median = cv2.blur(gray2,(25,25))
			height = img.shape[0]
			width = img.shape[1]
			size = img.size

			x = max(height/800 + 1, width/800 + 1)
			x = int(x)
			if(x%2==0):
				x=x-1
			kernel_size_erosion = x*4 + 1                                                                       # hyperparameter
			kernel_size_dilation = (x+1)*4 + 1                                                                  # hyperparameter

			kernel_dilate = np.ones((kernel_size_dilation, kernel_size_dilation),np.uint8)
			median = cv2.dilate(median,kernel_dilate,iterations = 2)


			#Threshold the image
			ret, thresh3 = cv2.threshold(median,
								127, 255, cv2.THRESH_BINARY)

			# cv2.imshow('window',gray2)
			# cv2.waitKey(0)
			#Contours application to the image containing panels only.

			contours2,hierachy,=cv2.findContours(thresh3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			cv2.imwrite("/home/galactica/Galactica/Production/galactica_solar_inspection/solar_inspection/media/thresh_img.jpg",thresh3)



			#Initialize the count variable to count the number of panels.
			#fname=os.path.splitext(img_name[j])[0]
			count=0
			for c in contours2:
				#Complete boxes to the contours which have created incomplete boxes.
				peri = cv2.arcLength(c, True)
				approx = cv2.approxPolyDP(c, 0.00001 * peri, True)
				#Add a bounding rectangle to the contours
				if(rotation==0):
					x,y,w,h = cv2.boundingRect(approx)
				#Use the height and width of the bounding box to remove unwanted boxes in the background
					if (10000 < w*h < (height*width)/8 ):
						count+=1
						points.append((x,y,x+w,y+h))
				else:
					rect = cv2.minAreaRect(approx)
					if (15 < w < 90 and 15 < h < 90):
						box = cv2.boxPoints(rect)
						box = np.int0(box)
						#cv2.drawContours(stitchimg,[box],0,(0,255,0),2)

			dict_img['Count'] = count
			dict_img['Points'] = points


		return dict_img
