import numpy
import cv2
import glob
import os
from skimage import measure
import imutils
from .Unet import Unet
from .IR_Segmentation import *
import numpy as np

class Segmentation_mask:
	"""
	Input: Images or List of Images

	Output: return mask of segmented panels

	Functions:

		deep_model(): Need to set up (RGB/IR), will return binary masks

		IP(): Uses Image processing to generate binary masks (RGB only), returns a list of masks

	"""
	def __init__(self, model_type,settings):
		# Initialize Unet

		self.unet = Unet(model_type, settings) #shared model RGB

	def unet_segmentation(self, data=[], resize = True, scale = True):
		# Call segmentation model get mask_list as return
		self.img_list = []
		if type(data) == list:
			self.img_list = data
		elif type(data) == numpy.ndarray:
			self.img_list.append(data)
		else :
			assert False, ('Wrong type of input')

		ret = self.unet.predict(self.img_list, resize, scale);
		masks = []
		for mask in ret:
			mask = mask *255;
			mask = np.uint8(mask)
			masks.append(mask)

		return masks

	def get_kernel_size(self, height, width):
		x = max(height/800 + 1, width/800 + 1)
		x = int(x)
		if(x%2==0):
			x=x-1
		return x


	def IR_segmentation(data):
		self.img_list = []
		if type(data) == list:
			self.img_list = data
		elif type(data) == numpy.ndarray:
			self.img_list.append(data)
		else :
			assert False, ('Wrong type of input')

		masks = []

		for img in self.img_list:
			mask = segment(img)
			masks.append(mask)
		return masks;


	def color_segmentation(self, data, img_type="Unet"):
		# use metadata to check whether image is RGB or not
		# release assert if not RGB

		#Iterate over images list for segmentation and counting.
		masks = []
		self.img_list = []
		if type(data) == list:
			self.img_list = data
		elif type(data) == numpy.ndarray:
			self.img_list.append(data)
		else :
			assert False, ('Wrong type of input')


		for j in range(0,len(self.img_list)):

			img = self.img_list[j].copy()
			rgbimg = self.img_list[j].copy()
			stitchimg = self.img_list[j].copy()

			height = img.shape[0]
			width = img.shape[1]

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

			k_size = self.get_kernel_size(height, width)

			#Morph_Erode
			kernel_erode = numpy.ones((k_size,k_size),numpy.uint8)
			erosion = cv2.erode(res,kernel_erode,iterations = 1)

			#cv2.imwrite('media/eroded{}.jpg'.format(j),erosion)

			#Morph_Dilate
			kernel_dilate = numpy.ones((31,31),numpy.uint8)
			dilation = cv2.dilate(res,kernel_dilate,iterations = 1)

			#cv2.imwrite('media/dilation{}.jpg'.format(j),dilation)

			#Convert dilated image to single channel to threshold
			gray = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)

			ret,thresh2 = cv2.threshold(gray,0,127,cv2.THRESH_OTSU)

			#cv2.imwrite('media/thresh2{}.jpg'.format(j),thresh2)

			#threshold the image and obtain the binary image
			(thresh3, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

			#cv2.imwrite('media/threh3{}.jpg'.format(j),im_bw)

			# #Morph_Erdoe
			# kernel_erode_2 = numpy.ones((5,5),numpy.uint8)
			# erosion_2 = cv2.erode(im_bw,kernel_erode_2,iterations = 2)
			#
			# cv2.imwrite('media/5erosion_2{}.jpg'.format(j),erosion_2)
			#
			#
			# #Morph_Dilate
			# kernel_dilate_2 = numpy.ones((7,7),numpy.uint8)
			# dilation = cv2.dilate(rgbimg,kernel_dilate_2,iterations = 1)
			#
			# cv2.imwrite('media/6_{}.jpg'.format(j),dilation)
			#
			# #Impose the binary mask on the original image
			# gray2 = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
			# erosion_2 = erosion_2[:,:,np.newaxis]
			# new = erosion_2 * rgbimg
			# cv2.imwrite('media/8_{}.jpg'.format(j),new)
			# #Threshold the image
			# ret, thresh3 = cv2.threshold(new, 127, 255, cv2.THRESH_BINARY)
			# cv2.imwrite('media/7_{}.jpg'.format(j),thresh3)


			masks.append(im_bw)

		return masks

	def reset(self):
		#Reset Keras model and free up memory
		self.unet.reset_keras();
