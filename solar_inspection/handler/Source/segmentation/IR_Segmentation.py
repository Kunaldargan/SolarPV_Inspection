import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from sklearn.mixture import GaussianMixture
import math
import os

np.set_printoptions(threshold=np.inf)

def show_image(img, name):
	cv2.imshow(name,img)
	cv2.waitKey(0)


def histogram(img):
	hist = cv2.calcHist([img],[0],None,[256],[0,256])
	plt.subplot(111),plt.plot(hist)
	plt.xlim([0,256])
	plt.show()

def threshold(img,val):
	ret, image = cv2.threshold(img,val,255,cv2.THRESH_BINARY)
	return image

def Morphology_open(img,kernel,iter=1):
	return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel,iterations=iter)

def Morphology_close(img,kernel,iter=1):
	return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel,iterations=iter)


def subtract(img,val):

	img2 = img.copy()
	img = img - val
	img [ img > img2] = 0
	return img


def get_good_contours(contours,perc):

	l = len(contours)

	if(l==1):
		return contours

	# sort contours in descending order based on area
	contours = sorted(contours,key=cv2.contourArea, reverse=True)

	# print_contours(contours)

	b_idx = l
	max_area = cv2.contourArea(contours[0])*perc/100
	for i in range(0,l):
		# current change
		# curr_change = (cv2.contourArea(contours[i])-cv2.contourArea(contours[i+1]))/(cv2.contourArea(contours[i]))

		if(cv2.contourArea(contours[i]) < max_area):
			b_idx = i
			break

	contours = contours[:b_idx]
	# print (b_idx)
	return contours


def print_contours(contours):
	print(len(contours))

	for c in contours:
		print(cv2.contourArea(c))



def watershed(img,sure_fg,sure_bg):


	sure_fg = np.uint8(sure_fg)
	sure_bg = np.uint8(sure_bg)

	img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
	img = np.uint8(img)

	unknown = cv2.subtract(sure_bg,sure_fg)

	ret, markers = cv2.connectedComponents(sure_fg)
	markers = markers+1
	markers[unknown==255] = 0
	markers = cv2.watershed(img,markers)
	img[markers == -1] = [255,0,0]

	return img, markers


def get_kernel(i,j):
	return np.ones((i,j),np.uint8)

def get_hist(img):

	# get histogram
	hist = cv2.calcHist([img],[0],None,[256],[0,256])
	hist = hist.astype(int)
	return hist

def show_hist(hist):
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.plot(hist)
	plt.xlim([0,256])
	plt.show()


def get_threshold(img):
	# get Data
	data = img.ravel()

	# Fit GMM
	gmm = GaussianMixture(n_components=2)												# HYPERPARAMETER
	gmm = gmm.fit(X=np.expand_dims(data,1))

	# Evaluate GMM
	gmm_x = np.linspace(0,255,256)

	gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1,1)))
	means = gmm.means_
	covariances = gmm.covariances_
	# print(means)
	# print(covariances)

	idx = 0
	max_mean = 0
	for i in range(0,means.shape[0]):
		if (means[i][0]>max_mean):
			max_mean = means[i][0]
			idx = i

	return max(int(means[idx] - (4*math.sqrt(covariances[idx][0]))),0)


def normalise(img,perc):

	# img = img.astype(np.int16)
	hist = get_hist(img)
	h,w = img.shape

	perc_pixels = h*w*perc/100
	curr_pixels = 0

	# prepare array for fk and gk
	fk =[]
	gk = []
	fk.append(0)
	gk.append(0)
	Dict = {}
	Dict[0]=0

	for i in range(0,256):
		if (hist[i] + curr_pixels < perc_pixels):
			curr_pixels+=hist[i]
		else:
			fk.append(i-1)
			curr_pixels = hist[i]

	if(fk[-1]!=255):
		fk.append(255)

	gk = np.linspace(0,255,len(fk),endpoint=True)

	for i in range(0, len(gk)):
		gk[i]=int(gk[i])


	for i in range(0,len(fk)-1):

		for f in range(fk[i]+1,fk[i+1]+1):
			Dict[f] = int(((f-fk[i])/(fk[i+1] - fk[i]) * (gk[i+1] - gk[i])) + gk[i])


	new_img = np.zeros(img.shape[:2],dtype="uint8")
	for i in range(0,h):
		for j in range(0,w):
			new_img[i,j] = Dict[img[i,j]]

	return new_img

################## CODE STARTS FROM HERE ####################
def segment(img) :

	# STEP 1 -> READ ORIGINAL
	h,w = img.shape
	hist = get_hist(img)

	# STEP 2 -> GET NORMALIZED
	T_norm = normalise(img.copy(),2)

	# STEP 3 -> THRESHOLD, BY FITTING GAUSSIAN DISTRIBUTION
	a = get_threshold(T_norm)
	T_norm_thresh = subtract(T_norm.copy(),a)

	# STEP 4 -> CALCULATE VARIANCE
	# add padding in image
	padded_image = cv2.copyMakeBorder(T_norm, 1,1,1,1,cv2.BORDER_CONSTANT, 0)
	h,w = padded_image.shape
	cnt = 0

	# variance map
	T_var = img.copy()
	sub_arr = np.zeros(9)
	for i in range(0,h-2):
		for j in range(0,w-2):
			idx=0

			# get subarray
			for ii in range(i, i+3):
				for jj in range(j, j+3):
					sub_arr[idx] = padded_image[ii,jj]
					idx+=1

			mean,stdDev = cv2.meanStdDev(sub_arr)
			variance = stdDev[0][0]#*stdDev[0][0]
			T_var[i,j]= int(variance)
	# show_image(T_var,'4.Variance_map, T_var')


	# STEP 5 -> HISTOGRAM EQUALIZATION OF VARIANCE MAP
	T_eqvar = cv2.equalizeHist(T_var)
	# show_image(T_eqvar,'5.Histogram equalized, T_eqvar')


	# STEP 6 -> THRESHOLD THE EQUALIZED HISOTGRAM																- HYPERPARAMETER
	# default - 2/3 of range, i.e 2*85
	Binarized_T_eqvar = threshold(T_eqvar,85)
	# show_image(Binarized_T_eqvar,'6.Thresholded, Binarized_T_eqvar')


	# STEP 7 -> NORMALIZED TEMPERATURE MAP (T_norm) WITHOUT IT'S VARIANCE
	# invert the Binarized_T_eqvar
	Binarized_T_eqvar_invert = cv2.bitwise_not(Binarized_T_eqvar)
	# apply it on T_norm_thresh
	T_norm_without_variance = cv2.bitwise_or(T_norm_thresh, T_norm_thresh, mask=Binarized_T_eqvar_invert)
	# show_image(T_norm_without_variance,'7.1.Normalized without variance, T_norm_without_variance')
	T_norm_without_variance_2 = threshold(T_norm_without_variance,1)
	# show_image(T_norm_without_variance_2,'7.2.Threshold the T_norm_without_variance, T_norm_without_variance_2')


	# STEP 8 -> USE WATERSHED TO FIND ARRAYS																	- HYPERPARAMETERS
	test0 = Morphology_close(T_norm_without_variance_2,get_kernel(3,3))
	# show_image(test0,'test0')

	# remove small noise
	test1 = Morphology_open(test0,get_kernel(7,7))
	# show_image(test1,'test1')

	# create bg mask
	test2 = cv2.dilate(test1,get_kernel(7,7))
	# show_image(test2,'test2')

	# closing in bg mask
	test3 = Morphology_close(test2,get_kernel(7,7))
	# show_image(test3,'test3')

	# remove small contours from bg mask
	contours, hierarchy = cv2.findContours(test3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	contours = get_good_contours(contours,10)																	# HYPERPARAMETER
	bg_mask = np.zeros(test3.shape[:2],dtype="uint8")
	bg_mask = cv2.drawContours(bg_mask,contours,-1,(255),-1)
	# show_image(bg_mask,'reduced_contoursxxx')

	bg_mask = cv2.dilate(bg_mask,get_kernel(7,7))
	# show_image(bg_mask,'after dilation')

	bg_mask = Morphology_close(bg_mask,get_kernel(7,7))
	# show_image(bg_mask,'closing2')

	# create fg mask
	fg_mask = cv2.erode(bg_mask,get_kernel(7,7),iterations=3)
	# show_image(fg_mask,'fg_mask')

	# apply watershed
	watersheded_image, markers = watershed(T_norm,fg_mask,bg_mask)
	# show_image(watersheded_image,'8.watersheded_image')


	# STEP 9 -> MAKE CONTOURS FROM WATERSHED OUTPUT

	# convert markers to binary image
	markers[markers > 1] = 255
	markers[markers <= 1] = 0
	markers = markers.astype(np.uint8)
	# show_image(markers,'9.markers')


	# STEP 10 -> FIND LARGEST CONTOUR

	# find contours
	contours, hierarchy = cv2.findContours(markers, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	c = max(contours, key = cv2.contourArea)
	Largest_contour = markers.copy()
	cv2.drawContours(Largest_contour, [c], 0, (150), 3)
	# show_image(Largest_contour,'10.Largest contour')


	# STEP 11 -> FIT ELLIPSE AND ROTATE

	# fitted ellipse
	ellipse = cv2.fitEllipse(c)
	cv2.ellipse(Largest_contour,ellipse,(50),2)
	# show_image(Largest_contour,'11.1.Fitted ellipse')

	# ellipse rotation
	# print(len(ellipse))
	# print((ellipse[0]))  # (x,y)
	# print((ellipse[1]))  # (Ma,ma)
	# print((ellipse[2]))  # angle

	# rotate image
	rotated_img = imutils.rotate_bound(img, 90-ellipse[2])
	# show_image(rotated_img,'11.2.Rotated Image')

	rotated_img_norm = imutils.rotate_bound(T_norm, 90 - ellipse[2])



	# STEP 12 -> ROTATE MARKER AND FIT rectangles
	# rotate marker
	markers = imutils.rotate_bound(markers,90-ellipse[2])

	# bound rectangles in marker
	contours, hierarchy = cv2.findContours(markers, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	rectangles = []

	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		rectangles.append(np.array([[x,y],[x,y+h],[x+w,y+h],[x+w,y]]))

	# create blank mask and fill rectangles
	blank = np.zeros(markers.shape[:2],dtype="uint8")
	for rect in rectangles:
		cv2.fillPoly(blank,pts =[rect],color=(255))


	#rotate blank
	blank = imutils.rotate_bound(img, (90 - ellipse[2]))
	return blank;
