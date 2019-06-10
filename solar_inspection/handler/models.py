from django.db import models

class File(models.Model):
    file = models.FileField(blank=False, null=False)
    def __str__(self):
        return self.file.name

# class Image(models.Model):
# 	"""docstring for Image"""
# 	image = models.ImageField(blank=False, null=False)
# 	def __init__(self):
# 		return self.image.name
		
# image = models.ImageField(upload_to='/Users/aaa/Desktop/Rajat/DjangoApp/galactica_solar_inspection/solar_inspection/media')
# class Image(models.Model):
# 	property_id = models.ForeignKey(
# 	                'properties.Address',
# 	                null=False,
# 	                default=1,
# 	                on_delete=models.CASCADE
# 	            )
