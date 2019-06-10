import numpy
import cv2
import glob
import os
from skimage import measure
import imutils
class Segmentations:
	"""
	Inumpyut: Images or List of Images

	Output: Json file with containing segmented out solar panels - points, counts and ordered set

	Functions: 
		segment() : Returns background segmentation points, count of panels and arrays in dictionary
		order_by() : to get order of solar panels, returns dictionary
		array_count(): to get solar arrays counts, returns dictionary
		panel_count(): to get solar panel counts, returns dictionary
	"""

	def __init__(self, data):
		self.segments = {}
		if type(data) == list:
			self.img_list = data
		elif type(data) == numpy.ndarray:
			self.img_list = list(data)
		else :
			assert False, ('Wrong type of inumpyut')

	def segment(self) :
		#Iterate over a list of images to detect segments
		for j in range(0,(len(self.img_list))):

			self.segments[j] = {}
			#check out fault types
			self.segments[j]['Counts'] = self.count_panel(self.img_list[j])

		return self.segments


	def count_panel(self, img):

		rotation = 0
		points =[]
		dict_img = {}
		img = img.copy()
		rgbimg = img.copy()
		stitchimg = img.copy()
		
		#Morph_Open
		kernel_opening = numpy.ones((11,11),numpy.uint8)
		img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_opening)

		#Convert image to single channel and change color colding to HSV
		hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		#Threshold blue color only to segment the panels from HSV image obtained
		COLOR_MIN = numpy.array([110,50,50],numpy.uint8)  
		COLOR_MAX = numpy.array([130,255,255],numpy.uint8)
		frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
		
		#apply mask to the binary image o view panels.
		mask = frame_threshed
		res = cv2.bitwise_and(img,img, mask= mask) 

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
		ret, thresh3 = cv2.threshold(new,
						127, 255, cv2.THRESH_BINARY)

		#Contours application to the image containing panels only. 
		contours2,hierachy,=cv2.findContours(thresh3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

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
				if (20 < w < 90 and 20 < h < 90):
					#cv2.rectangle(stitchimg,(x, y), (x+w, y+h), (0, 255, 0), 2)
					count+=1
				points.append((x,y,x+w,y+h))
			else:
				rect = cv2.minAreaRect(approx)
				if (15 < w < 90 and 15 < h < 90):
					box = cv2.boxPoints(rect)
					box = numpy.int0(box)
					#cv2.drawContours(stitchimg,[box],0,(0,255,0),2) 

		dict_img['Count'] = count
		dict_img['Points'] = points     
		
		
		return dict_img

	
