import numpy
import cv2
import glob
import os
from skimage import measure
import imutils

class segmentation_mask:
	"""
	Input: Images or List of Images 

	Output: return mask of segmented panels

	Functions: 
		
		deep_model(): Need to set up (RGB/IR), will return binary masks

		IP(): Uses Image processing to generate binary masks (RGB only), returns a list of masks

	"""
	def __init__(self, data):
		if type(data) == list:
			self.img_list = data
		elif type(data) == numpy.ndarray:
			self.img_list = list(data)
		else :
			assert False, ('Wrong type of input')



	# def deep_model(self):
		# need to implement


	def IP(self):
		# use metadata to check whether image is RGB or not
		# release assert if not RGB

		#Iterate over images list for segmentation and counting. 
		masks = []

		for j in range(0,len(self.img_list)):

			img = self.img_list[j].copy()
			rgbimg = self.img_list[j].copy()
			stitchimg = self.img_list[j].copy()
			#Morph_Open
			kernel_opening = numpy.ones((11,11),numpy.uint8)
			img = cv2.morphologyEx(self.img_list[j], cv2.MORPH_OPEN, kernel_opening)

			#Convert image to single channel and change color colding to HSV
			hsv_img = cv2.cvtColor(self.img_list[j], cv2.COLOR_BGR2HSV)

			#Threshold blue color only to segment the panels from HSV image obtained
			COLOR_MIN = numpy.array([110,50,50],numpy.uint8)  
			COLOR_MAX = numpy.array([130,255,255],numpy.uint8)
			frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
			
			#apply mask to the binary image o view panels.
			mask = frame_threshed
			res = cv2.bitwise_and(self.img_list[j],self.img_list[j], mask= mask) 

			#Morph_Erode 
			kernel_erode = numpy.ones((7,7),numpy.uint8)
			erosion = cv2.erode(res,kernel_erode,iterations = 1)
			
			#Morph_Dilate 
			kernel_dilate = numpy.ones((31,31),numpy.uint8)
			dilation = cv2.dilate(erosion,kernel_dilate,iterations = 1)

			#Convert dilated image to single channel to threshold
			gray = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)

			ret,thresh2 = cv2.threshold(gray,0,127,cv2.THRESH_OTSU)

			#threshold the image and obtain the binary image
			(thresh3, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

			#Morph_Erdoe
			kernel_erode_2 = numpy.ones((5,5),numpy.uint8)
			erosion_2 = cv2.erode(im_bw,kernel_erode_2,iterations = 2)

			#Morph_Dilate
			kernel_dilate_2 = numpy.ones((7,7),numpy.uint8)
			dilation = cv2.dilate(rgbimg,kernel_dilate_2,iterations = 1)

			#Impose the binary mask on the original image
			gray2 = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
			new = erosion_2 * gray2
			#Threshold the image 
			ret, thresh3 = cv2.threshold(new, 127, 255, cv2.THRESH_BINARY)

			masks.append(thresh3)
			print (type(masks))

		return masks
