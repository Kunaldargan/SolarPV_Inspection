from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.renderers import JSONRenderer

import cv2
from PIL import Image
import json
import io
import numpy as np
import time
import os

from django.http import HttpResponse
from django.http import QueryDict
from django.http import JsonResponse

from .serializers import FileSerializer
from .Source.fault_detection.faults import Faults
from .Source.segmentation.counts import Counts
from .Source.segmentation.segmentation_mask import Segmentation_mask
from .Source.stitching.stitch import stitch
from .Source.exiftool.exiftool import Extract_Exif
from .Source.Utils.utils import process_mask, plot_segments
from .app_settings import Settings_Parser
from solar_inspection.settings import BASE_DIR


class FileUploadView(APIView):
	parser_classes = (MultiPartParser, FormParser)

	def get(self, request):
		all_images = Image.objects.all()
		serializer = ImageSerializer(all_images, many=True)
		return JsonResponse(serializer.data, safe=False)

	def post(self, request, *args, **kwargs):

		settings = Settings_Parser()
		# # converts querydict to original dict
		images = dict((request.data).lists())['file']
		flag = 1
		arr = []
		for i in range(len(images)):
			dict_imgs = {'file': images[i] }
			file_serializer = FileSerializer(data=dict_imgs)

			if file_serializer.is_valid():
				file_serializer.save()
				arr.append(file_serializer.data)
			else:
				flag = 0


		# create directory for current timestamp

		timestr = time.strftime("%Y%m%d-%H%M%S")
		os.mkdir('media/'+timestr)
		media_path = 'media/'+str(timestr)+'/'
		fault_data = {}
		segmentation_data = {}
		img_list = []
		e = Extract_Exif()
		# exif_data = e.Extract_MetaData('/Users/aaa/Downloads/New Folder With Items/35.jpg')
		path_list = [];

		for img_path in arr:
			path = BASE_DIR + img_path['file'];
			img_name = os.path.basename(img_path['file']);
			img = cv2.imread(path,cv2.IMREAD_UNCHANGED)
			#append to image list
			img_list.append(img)
			path_list.append(path)

		# get exif data
		exif_data = e.Extract_MetaData(path_list)
		# get IR or RGB
		color_components = exif_data[img_name]['File']['ColorComponents']
		print(color_components)

		if (color_components==1): # IR

			# run deep model and get mask
			obj = Segmentation_mask("COM", settings)
			masks = obj.unet_segmentation(img_list)

			for i in range(len(img_list)):
				mask = masks[i].copy()
				img = img_list[i].copy()
				Image_Height, Img_Width = img.shape[:2]

				cv2.imwrite(os.path.join(media_path,'1_mask{}.jpg'.format(i)),mask)

				mask_out = cv2.resize(mask, dsize=(Image_Height, Img_Width),interpolation=cv2.INTER_NEAREST)
				cv2.imwrite(os.path.join(media_path,'2_mask_reiszed{}.jpg'.format(i)),mask_out)

				# apply on image
				#seg_image = cv2.bitwise_and(img,img, mask=mask)

				# color_segmentation
				#col_segment = obj.color_segmentation()

				# count panels

				# ordering decide

				# fault detection
				#fault_data_final[img_name]=Faults(col_segment).classify()

		elif (color_components==3): # RGB
			# run deep model and get mask
			obj = Segmentation_mask("RGB", settings)
			masks = obj.unet_segmentation(img_list)
			count_obj = Counts()

			for i in range(len(img_list)):

				# mask = masks[i].copy()
				# img = img_list[i].copy()
				# Image_Height, Img_Width = img.shape[:2]
				#
				# mask_out = cv2.resize(mask, dsize=(Img_Width, Image_Height),interpolation=cv2.INTER_NEAREST)
				#
				# mask = process_mask(mask_out,i,timestr)
				# cv2.imwrite(os.path.join(media_path,'mask_processed{}.jpg'.format(i)),mask)
				#
				# mask = mask[:,:,np.newaxis]
				# res = cv2.bitwise_and(img,img,mask = mask)
				# cv2.imwrite(os.path.join(media_path,'masked_apply_{}.jpg'.format(i)),res)

				# color segmentation
				results = obj.color_segmentation(img_list[i])
				m = results[0][:,:,np.newaxis]
				output = cv2.bitwise_and(img,img,mask = m)
				cv2.imwrite(os.path.join(media_path,'color_segm{}.jpg'.format(i)),output)

				# count panels
				segmentation_data[img_name] = count_obj.count_panel(output)

				# odering decide

				# recreate segments from exif_data
				counts_img = plot_segments(img, segmentation_data[img_name])
				cv2.imwrite(os.path.join(media_path,'counts_plotted_img_{}.jpg'.format(i)),counts_img)

		response = {**fault_data, **segmentation_data}

		if flag == 1:
			return Response(response, status=status.HTTP_201_CREATED)
		else:
			return Response(0, status=status.HTTP_400_BAD_REQUEST)
