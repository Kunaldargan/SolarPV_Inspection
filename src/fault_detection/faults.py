import numpy
import cv2
import glob
import os
from skimage import measure
import imutils
class Faults:
	"""
	Input: Images or List of Images

	Output: Json file with information about faults and their location

	Functions:
		classify(): classifies in multiple classes of faults, returns fault dictionary for image list
		hotspots(): to get hotspots, returns dictionary 
		hotpatches(): to get hotpatches, returns dictionary
	"""
	def __init__(self, data):
		self.faults = {}
		if type(data) == list:
			self.img_list = data
		elif type(data) == numpy.ndarray:
			self.img_list = list(data)
		else :
			assert False, ('Wrong type of input')
				
	def classify(self) :
		#Iterate over a list of images to detect faults
		for j in range(0,(len(self.img_list))):

			self.faults[j] = {}
			#check out fault types

			self.faults[j]['hotspots'] = self.hotspots(self.img_list[j])

			self.faults[j]['hotpatches'] = self.hotpatches(self.img_list[j])


		return self.faults


	def hotspots(self, img):

		points = []
		#Convert image to single channel
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#Blur image to reduce noise 
		blurred = cv2.GaussianBlur(gray, (3,3), 0)
		#Threshold the image over a set of values
		thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)[1]
		#From the thresholded image pick only blobs which have specified neighbors.
		labels = measure.label(thresh, neighbors=4, background=0)
		#Create a black mask where the blobs obatined in labels can be pasted
		mask = numpy.zeros(thresh.shape, dtype="uint8")

		for label in numpy.unique(labels):
			labelMask = numpy.zeros(thresh.shape, dtype="uint8")
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
				points.append((int(cX), int(cY), float(radius)))
				# add cX, cY, radius in json depending on image
		
		return points


	def hotpatches(self, img):
		
		hot_dict = {'Diode_fault':{}, 'String_fault':{} }

		#Convert image to sngle channel image
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		#Blur the image to reduce noise
		blurred = cv2.GaussianBlur(gray, (3,3), 0)
		
		#Threshold the image over a particular range of pixel values
		thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
		
		#Morph_Dilate to bring out the pixels
		thresh = cv2.dilate(thresh, None, iterations=4)
		
		#From the thresholded image pick only blobs which have specified neighbors.
		labels = measure.label(thresh, neighbors=4, background=0)
		
		#Create a black mask where the blobs obatined in labels can be pasted
		mask = numpy.zeros(thresh.shape, dtype="uint8")

		for label in numpy.unique(labels):
			if label == 0:
				continue

			labelMask = numpy.zeros(thresh.shape, dtype="uint8")
			labelMask[labels == label] = 255
			numPixels = cv2.countNonZero(labelMask)
			#Paste the blob only if the value of pixels in it is above a certain threshold value
			if numPixels > 300:
				mask = cv2.add(mask, labelMask)
		#Apply contours on the mask image which contains the blobs
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		#Contours are applied and the Convex hull is drawn on the image from left to right.
		cnt1=0
		cnt2=0
		for (i, c) in enumerate(cnts):
			area = cv2.contourArea(c)
			#(x, y, w, h) = cv2.boundingRect(c)
			#((cX, cY), radius) = cv2.minEnclosingCircle(c)
			hull = cv2.convexHull(c)
			hull = numpy.squeeze(hull, axis = 1)

			#Draw convex hull on the blobs of the mask image depending on the area it is a diode fault or string fault
			if (area<1000):#assumption
				
				#cv2.drawContours(cv_img[j], [hull], -1, (0, 255, 0), 3)                    
				#cv2.putText(cv_img[j], "Diode Fault", (x+20, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
				hot_dict['Diode_fault'][cnt1] = hull
				cnt1+=1
			else:
				#cv2.drawContours(cv_img[j], [hull], -1, (0, 255, 0), 3)              
				#cv2.putText(cv_img[j], "String Fault", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
				hot_dict['String_fault'][cnt2] = hull
				cnt2+=1

		return hot_dict



