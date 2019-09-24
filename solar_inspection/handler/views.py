from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.renderers import JSONRenderer
from .serializers import FileSerializer

import time
import os

from django.http import HttpResponse
from django.http import QueryDict
from django.http import JsonResponse

#parse App Settings
from .app_settings import Settings_Parser

#Main Handler
from .handler import Handler

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
		list_of_paths = []
		for i in range(len(images)):
			dict_imgs = {'file': images[i] }
			file_serializer = FileSerializer(data=dict_imgs)

			if file_serializer.is_valid():
				file_serializer.save()
				list_of_paths.append(file_serializer.data)
			else:
				flag = 0


		# create directory for current timestamp
		timestr = time.strftime("%Y%m%d-%H%M%S")
		os.mkdir(os.path.join(settings.media_path,timestr))
		media_path_timestamp = os.path.join(settings.media_path,timestr)

		handler = Handler(settings, media_path_timestamp);
		handler.extract_metadata(list_of_paths);
		handler.solar_inspection_main();

		fault_data={}
		segmentation_data={}
		response = {**fault_data, **segmentation_data}

		if flag == 1:
			return Response(response, status=status.HTTP_201_CREATED)
		else:
			return Response(0, status=status.HTTP_400_BAD_REQUEST)
