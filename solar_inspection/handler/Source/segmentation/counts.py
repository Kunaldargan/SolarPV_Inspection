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

	def count_panel_contours(self, img, panel_type="Drapper"):

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
			dict_img['Count'] = count
			dict_img['Points'] = points
		return dict_img

	def rotated_rect_mask(self, img, mask,contours): #incomplete : to be added counting
		cnts = []
		bbox = []

		for (i, c) in enumerate(contours):
			area = cv2.contourArea(c)
			rect = cv2.minAreaRect(c)
			xy, wh, a = rect # a - angle

			x,y = xy
			w,h = wh
			print(type(x))
			if (w > im.shape[0]*0.2 and h>50):
				box = cv2.boxPoints(rect)
				box = np.int0(box) #turn into ints
				cv2.drawContours(mask,[box],0,(0,255,0),3)
				cv2.drawContours(black,[box],0,255,thickness=cv2.FILLED)
				cv2.fillPoly(mask, pts =[c], color=(255,255,255))

				cnts.append(c)
				bbox.append((x, y, w, h))
		img = cv2.bitwise_and(img,img, mask = black)
		return img, mask, cnts, bbox

	def rect_mask(self, img, mask, contours):
		c = max(contours, key = cv2.contourArea)
		ellipse = cv2.fitEllipse(c)
		black = np.zeros((img.shape[0], img.shape[1]), np.uint8)
		rotated_img = imutils.rotate_bound(img, ellipse[2])
		rotated_mask = imutils.rotate_bound(mask, ellipse[2])
		black = imutils.rotate_bound(black, ellipse[2])

		blurred = cv2.GaussianBlur(rotated_mask, (5,5), 0)

		contours, hierarchy = cv2.findContours(blurred,
			cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		cnts = []
		bbox = []
		count = 0;
		row = 0;
		for (i, c) in enumerate(contours):
			area = cv2.contourArea(c)
			x,y,w,h = cv2.boundingRect(c)

			if (w > img.shape[0]*0.2 and h>50):

				cv2.drawContours(rotated_mask,c,0,(0,255,0),3)
				cv2.rectangle(black, (x,y), (x+w, y+h),255,thickness=cv2.FILLED)
				cnts.append(c)
				bbox.append((x, y, w, h))

		print("total cells", count)
		rotated_img = cv2.bitwise_and(rotated_img,rotated_img, mask = black)

		for x,y,w,h in bbox:
			yr = int(y+h*0.5)
			cv2.rectangle(rotated_img, (x,y), (x+w, y+h),(0,255,0),thickness=3)
			cv2.rectangle(rotated_img, (x,y), (x+w, yr),(0,255,0), thickness=2)
			i = 0
			width = x+w
			row +=1
			cv2.putText(rotated_img," row= "+str(row)+": col= "+str(count+1),(x,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(0,0, 255),3)
			while(x < width-20):
				cv2.rectangle(rotated_img, (x,y), (x+20, y+h),(0,255,0), thickness=3)
				count +=1
				x=x+20
			cv2.putText(rotated_img," row= "+str(row)+": col= "+str(count),(x,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(0,0, 255),3);

		return rotated_img, black, cnts, bbox

	def count_panel(self, data = [], masks = [], panel_width=20):
		if type(data) == list:
			self.img_list = data
		elif type(data) == numpy.ndarray:
			self.img_list = list(data)
		else :
			assert False, ('Wrong type of input')

		if type(masks) == list:
			self.mask_list = masks
		elif type(masks) == numpy.ndarray:
			self.mask_list = list(masks)
		else :
			assert False, ('Wrong type of masks')

		annotated_imgs = []
		extracted_masks = []
		for i in range(len(self.img_list)):
			im = self.img_list[i]
			cv2.imshow("window",im)
			cv2.waitKey(0)
			mask = self.mask_list[i]
			black = np.zeros((im.shape[0], im.shape[1]), np.uint8)
			blurred = cv2.GaussianBlur(mask, (5,5), 0)

			contours, hierarchy = cv2.findContours(blurred,
				cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
			cnts = []
			bbox = []

			img, mask_, cnts, bbox = self.rect_mask(im.copy(), mask.copy() ,contours)
			cv2.imshow("window",mask_)
			cv2.waitKey(0)
			annotated_imgs.append(img)
			extracted_masks.append(mask_)

		return annotated_imgs, extracted_masks;
