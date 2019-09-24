import os
import cv2
from matplotlib import pyplot as plt

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

img = cv2.imread("/home/galactica/Galactica/Datasets/Eric2/skip4/2019:03:26_20:33:36.jpg",0) #file:///home/galactica/Galactica/Datasets/Eric2/outputs_contrasting/2019:03:26_20:33:36.jpg
hist = get_hist(img)
cv2.imshow("auto",img)
cv2.waitKey(0)

show_hist(hist)
