import os
import shutil
from PIL import Image
from PIL.ExifTags import TAGS


def get_minimum_creation_time(exif_data):
    mtime = "?"
    if 306 in exif_data and exif_data[306] < mtime: # 306 = DateTime
        mtime = exif_data[306]
    if 36867 in exif_data and exif_data[36867] < mtime: # 36867 = DateTimeOriginal
        mtime = exif_data[36867]
    if 36868 in exif_data and exif_data[36868] < mtime: # 36868 = DateTimeDigitized
        mtime = exif_data[36868]
    return mtime

def print_all_exif_tags(image):
	try:
		img = Image.open(image)
	except (Exception, e):
		print(image, "skipping due to", e)
	else:
		xf = img._getexif()
		print(type(xf))
		print(xf.keys())
		print("DateTimeOriginal ** ", get_minimum_creation_time(xf))
	for tag in xf:
		print(TAGS.get(tag), xf[tag])

def sort_date_time(img_list, output_dir):
	result = []
	for img in img_list:
		im = Image.open(img)
		xf = im._getexif()
		mtime = str(get_minimum_creation_time(xf))
		result.append((mtime,img))
		out = sorted(result, key = lambda x:x[0])
	print(out)
	
	for data, img in result:
		data = data.replace(" ","_")
		shutil.copy(img, os.path.join(output_dir,data)+".jpg")
		

if __name__=="__main__":
	source_dir = raw_input("Source Dir : ") 
	output_dir = raw_input("Output Dir : ") 
	img_list = [os.path.join(source_dir,img) for img in os.listdir(source_dir)]	
	#print_all_exif_tags(img_list[0]) # set which key corresponds to datetime
	sort_date_time(img_list, output_dir)
	
	
