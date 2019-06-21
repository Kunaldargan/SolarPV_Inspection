import cv2
from solar_inspection.settings import BASE_DIR
import numpy as np

def process_mask(mask,i,timestr):
    """
        return enhanced image
    """

    # #ret, thresh = cv2.threshold(mask,5,255,cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,101,1)
    # cv2.imwrite('media/'+str(timestr)+'/2.1_adptive_threshold{}.jpg'.format(0),thresh)
    # #Closing to remove black spots
    # median = cv2.medianBlur(thresh,5)
    # cv2.imwrite('media/'+str(timestr)+'/2.2_median{}.jpg'.format(0),median

    # blur the Image
    median = cv2.blur(mask,(71,71))                                                                    # hyperparameter
    # cv2.imwrite('media/'+str(timestr)+'/2.1_median_blur{}.jpg'.format(i), median)

    # threshold
    ret, threshold = cv2.threshold(median,35,255,cv2.THRESH_BINARY)                                    # hyperparameter
    # cv2.imwrite('media/'+str(timestr)+'/2.2_thresholded{}.jpg'.format(i), threshold)

    height = mask.shape[0]
    width = mask.shape[1]
    x = max(height/800 + 1, width/800 + 1)
    x = int(x)
    if(x%2==0):
    	x=x-1
    kernel_size_erosion = x*4 + 1                                                                       # hyperparameter
    kernel_size_dilation = (x+1)*4 + 1                                                                  # hyperparameter


    kernel_erosion = np.ones((kernel_size_erosion, kernel_size_erosion),np.uint8)
    kernel_dilate = np.ones((kernel_size_dilation, kernel_size_dilation),np.uint8)
    kernel_dilate2 = np.ones((kernel_size_dilation*2, kernel_size_dilation*2),np.uint8)

    # erosion
    erosion = cv2.erode(threshold,kernel_erosion, iterations=1)
    dilation = cv2.dilate(erosion,kernel_dilate,iterations = 1)
    # cv2.imwrite('media/'+str(timestr)+'/2.3_erosion_mask{}.jpg'.format(i),erosion)
    # cv2.imwrite('media/'+str(timestr)+'/2.4_dilation_mask{}.jpg'.format(i),dilation)

        # remove small areas
        #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(dilation, connectivity=8)    # hyperparameter
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 1000000                                                                                      # hyperparameter

    #your answer image
    img2 = np.zeros((output.shape),dtype=np.uint8)
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255


    # cv2.imwrite('media/'+str(timestr)+'/2.5_removed_areas{}.jpg'.format(i),img2)

    # dilation
    dilation2 = cv2.dilate(img2, kernel_dilate2,iterations=3)                                                    # hyperparameter
    # cv2.imwrite('media/'+str(timestr)+'/2.6_dilated_areas{}.jpg'.format(i),dilation2)
    #closing = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel_1, iterations =2)
    #ret, closing = cv2.threshold(closing, 5,255,cv2.THRESH_BINARY)

    return dilation2

def plot_segments(img, segmentation_data):

    out = img.copy()

    for pts in segmentation_data['Points'] :
            cv2.rectangle(out, (pts[0],pts[1]), (pts[2],pts[3]), (0,255,0), 3)

    return out
