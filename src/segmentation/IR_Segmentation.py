#import numpy as np
#from PIL import Image
#im = Image.open('DJI_0024.tif')
#import tifffile as tiff
#a = tiff.imread('DJI_0024.tif')
import cv2
import numpy as np
imgcv = cv2.imread('DJI_0245.jpg')
img2 = cv2.imread('DJI_0245.jpg')
from matplotlib import pyplot as plt


hsv_img = cv2.cvtColor(imgcv, cv2.COLOR_BGR2HSV)
hsv_img2 = cv2.cvtColor(imgcv, cv2.COLOR_RGB2HSV)

hist = cv2.calcHist([hsv_img], [2], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])

#Threshold 

#cv2.imwrite('HSV.jpg',hsv_img)
#
#hsv_img5 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

### 85 upper 
######These array are BGR values ####Required function 
# 111 0 0        180 255 255   


COLOR_MIN = np.array([0,0,120],np.uint8)  ###upper circle bar
COLOR_MAX = np.array([255,255,255],np.uint8)    #### lower circle bar 
frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
cv2.imwrite('frame_threshold.jpg',frame_threshed)

mask = frame_threshed
res = cv2.bitwise_and(imgcv,imgcv, mask= mask) 
cv2.imwrite('Panels.jpg',res)

gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

ret,thresh1 = cv2.threshold(dst,50,255,cv2.THRESH_BINARY)

cv2.imwrite('thresholdopencv.png',thresh1)

mask = gradientimg

res = cv2.bitwise_and(gray,gray,mask=mask)

cv2.imwrite('Panels.png',res)


kernel2 = np.ones((11,11),np.uint8)
opening = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel2)
cv2.imwrite('op2.jpg',opening)

canny = cv2.Canny(opening,200,350)
cv2.imwrite('Cannyedgeop.jpg',canny)




_,contours,hierachy,=cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


img2 = cv2.imread('DJI_0245.jpg')


count=0
for c in contours:
    # get the bounding rect
#    peri = cv2.arcLength(c, True)
#    approx = cv2.approxPolyDP(c, 0.00001 * peri, True)
#    rect = cv2.minAreaRect(approx)
#    box = cv2.boxPoints(rect)
#    box = np.int0(box)
#    cv2.drawContours(img2,[box],0,(0,0,255),2)
#    count = count+1
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.00001 * peri, True)
    x,y,w,h = cv2.boundingRect(approx)
    if (20 < w < 100   and 20< h < 100):
        cv2.rectangle(img2,(x, y), (x+w, y+h), (0, 255, 0), 2)
        count+=1
print(count)
#cv2.imwrite('contoursblurtop2dilation.jpg',rgbimg)
cv2.imwrite('Thermal_Panorama_Count_2.jpg',img2)
#COunt : 118
#COunt :471 
#New count is 77 

