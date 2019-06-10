from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.renderers import JSONRenderer

import cv2
from PIL import Image
import json
import io
import numpy
import time
import os

from django.http import HttpResponse
from django.http import QueryDict
from django.http import JsonResponse

from .serializers import FileSerializer
from .Source.fault_detection.faults import Faults
from .Source.segmentation.segmentation import Segmentations
from .Source.segmentation.segmentation_mask import segmentation_mask
from .Source.stitching.stitch import stitch
from .Source.exiftool.exiftool import Extract_Exif

class FileUploadView(APIView):
	parser_classes = (MultiPartParser, FormParser)

	def get(self, request):
		all_images = Image.objects.all()
		serializer = ImageSerializer(all_images, many=True)
		return JsonResponse(serializer.data, safe=False)

	def post(self, request, *args, **kwargs):


		# # converts querydict to original dict
		
		images = dict((request.data).lists())['file']
		print (images)
		flag = 1
		arr = []
		i = 0
			
		for i in range(len(images)):
			dictt = {'file': images[i] }
			file_serializer = FileSerializer(data=dictt)

			if file_serializer.is_valid():
				file_serializer.save()
				arr.append(file_serializer.data)
			else:
				flag = 0


		# create directory for current timestamp

		timestr = time.strftime("%Y%m%d-%H%M%S")
		os.mkdir('media/'+timestr)
		path2 = os.getcwd()
		fault_data_final = {}
		segmentation_data_final = {}
		
		# do real stuff

		e = Extract_Exif()
		# exif_data = e.Extract_MetaData('/Users/aaa/Downloads/New Folder With Items/35.jpg')

		for i in arr:
			
			path = os.path.join(path2,i['file'])
			img_name = os.path.basename(i['file'])
			img = cv2.imread(path)
			img = numpy.expand_dims(img,axis=0)

			# get exif data
			exif_data = e.Extract_MetaData(path)
			# get IR or RGB
			color_components = exif_data[img_name]['File']['ColorComponents']
			
			if (color_components=='1'): # IR

				# run deep model and get mask
				mask = segmentation_mask(img).deep_model()

				# apply on image
				seg_image = mask * img

				# count panels


				# ordering decide


				# fault detection
				fault_data_final[img_name]=Faults(img).classify()

			elif (color_components=='3'): # RGB

				# run deep model and get mask
				mask = segmentation_mask(img).deep_model()

				# apply on image
				seg_image = mask * img

				# count panels
				segmentation_data_final[img_name]= Segmentations(seg_image).segment()

				# odering decide



		response = {**fault_data_final, **segmentation_data_final}


		if flag == 1:
			return Response(response, status=status.HTTP_201_CREATED)
		else:
			return Response(0, status=status.HTTP_400_BAD_REQUEST)