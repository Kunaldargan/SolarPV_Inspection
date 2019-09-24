import cv2
from PIL import Image
import json
import io
import numpy as np
import time
import os


# from .Source.fault_detection.faults import Faults
from .Source.fault_detection.faultDetection import FaultDetection
from .Source.segmentation.counts import Counts
from .Source.segmentation.segmentation_mask import Segmentation_mask
from .Source.stitching.stitch import stitch
from .Source.exiftool.exiftool import Extract_Exif
from .Source.Utils.utils import process_mask, plot_segments
from solar_inspection.settings import BASE_DIR

class Handler():

	def __init__(self,settings,path):
		self.fault_data = {}
		self.segmentation_data = {}
		self.Image_ = []
		self.Extract_Exif = Extract_Exif()
		self.path_list = [];
		self.settings = settings;
		# get IR or RGB
		self.Image_Height = self.settings.Image_Height
		self.Image_Width  = self.settings.Image_Width
		self.Image_Channels = self.settings.Channels #exif_data[img_name]['File']['ColorComponents']

		self.media_path_timestamp = path


	def extract_metadata(self,arr):
		for img_path in arr:
			path = BASE_DIR + img_path['file'];
			img_name = os.path.basename(img_path['file']);
			self.path_list.append(path)

		# get exif data
		self.exif_data = self.Extract_Exif.Extract_MetaData(self.path_list)
		return(self.exif_data)

	def solar_inspection_main(self,Mode="IR"):

		if (Mode == "IR"): # IR
			################### To be reviewed ############################################

			self.Image_ = [cv2.imread(path,1) for path in self.path_list ]

			# run solar array mask
			obj = Segmentation_mask("IR", self.settings)

			if self.settings.seg_mode == "DNN":
				masks = obj.unet_segmentation(self.Image_)
			elif self.settings.seg_mode == "IP":
				masks = obj.IR_segmentation(self.Image_)
			else :
				assert False, ('Segmentation can be through DNN or IP modes')

			for i, im_path in enumerate(self.path_list):
				mask = masks[i].copy()
				img = self.Image_[i].copy()
				file_name = im_path.split("/")[-1]
				self.Image_Height, self.Img_Width = img.shape[:2]

				mask_out = cv2.resize(mask, dsize=(self.Img_Width, self.Image_Height),interpolation=cv2.INTER_NEAREST)
				print("saving mask to path : ",os.path.join(self.media_path_timestamp,'mask_{}'.format(file_name)) )
				cv2.imwrite(os.path.join(self.media_path_timestamp,'mask_{}'.format(file_name)),mask_out)
			obj.reset(); # release mem
			faults = FaultDetection(self.settings);
			faults.fault_detect_valid(path_list = self.path_list, output_path = self.media_path_timestamp);

			# count panels
			# count_obj = Counts()
			# annotated_imgs, mask_dets = count_obj.count_panel(data = self.Image_, masks = masks, panel_width=20)

			# # fault detection IP
			# images, detections = faults.fault_detection(self.Image_);
			# for i, img in enumerate(images):
			# 	print("saving mask to path : ",os.path.join(self.media_path_timestamp,'annotation_{}.jpg'.format(str(i))) )
			# 	cv2.imwrite(os.path.join(self.media_path_timestamp,'annotation__{}.jpg'.format(str(i))),img)


		elif (Mode=="RGB"): # RGB

			self.Image_ = [cv2.imread(path,1) for path in self.path_list ]
			# run deep model and get mask
			obj = Segmentation_mask("RGB", self.settings)
			masks = obj.unet_segmentation(self.Image_)
			count_obj = Counts()

			for i in range(len(self.Image_)):

				# mask = masks[i].copy()
				# img = Image_[i].copy()
				# Image_Height, Img_Width = img.shape[:2]
				#
				# mask_out = cv2.resize(mask, dsize=(Img_Width, Image_Height),interpolation=cv2.INTER_NEAREST)
				#
				# mask = process_mask(mask_out,i,timestr)
				# cv2.imwrite(os.path.join(media_path_timestamp,'mask_processed{}.jpg'.format(i)),mask)
				#
				# mask = mask[:,:,np.newaxis]
				# res = cv2.bitwise_and(img,img,mask = mask)
				# cv2.imwrite(os.path.join(media_path_timestamp,'masked_apply_{}.jpg'.format(i)),res)

				# color segmentation
				results = obj.color_segmentation(self.Image_[i])
				m = results[0][:,:,np.newaxis]
				output = cv2.bitwise_and(img,img,mask = m)
				cv2.imwrite(os.path.join(self.media_path_timestamp,'color_segm{}.jpg'.format(i)),output)

				# count panels
				segmentation_data[img_name] = count_obj.count_panel(output)

				# odering decide

				# recreate segments from exif_data
				counts_img = plot_segments(img, segmentation_data[img_name])
				cv2.imwrite(os.path.join(self.media_path_timestamp,'counts_plotted_img_{}.jpg'.format(i)),counts_img)
