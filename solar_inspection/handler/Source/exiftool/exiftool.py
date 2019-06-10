import exiftool
import sys
import os
import json
import pprint


class Extract_Exif :
    # extensions supported
    img_extensions = ['JPEG','PNG','TIFF','JPG']

    # unwanted items
    unwanted = ["ExifToolVersion","ImageDescription"]

    def __init__(self):
        return
    
    def Extract_MetaData(self, img_path):
        img_extensions = Extract_Exif.img_extensions
        unwanted = Extract_Exif.unwanted
        listOfFiles = []
        listOfFiles.append(img_path)
        
        Exif_Result_dict = {}

        with exiftool.ExifTool() as et:
            metadata = et.get_metadata_batch(listOfFiles)
    
        for d in metadata:
            txt = d['File:FileName']
            keys_d = d.keys()
            newkeys = list() #[]
            newdict = dict() #{}
            for s in keys_d :
                if ':' in s:
                    l=s.split(':')
                    if (l[1] in unwanted) :
                        continue
                    if (l[0] not in newdict.keys()) :
                        subdict = {}
                        subdict.update({l[1] : d[s]})
                        newdict.update({l[0] : subdict})
                    else :
                        subdict = newdict[l[0]]
                        subdict.update({l[1] : d[s]})
                        newdict.update({l[0] : subdict})
                else:
                    newdict.update({s : d[s]})
           
            Exif_Result_dict.update({txt : newdict})
            

        return Exif_Result_dict

""" 
Test locally

e = Extract_Exif()
print(e.Extract_MetaData('<directory path>'))
initlaize class        
e = Extract_Exif()
exif_data = e.Extract_MetaData('/Users/aaa/Downloads/New Folder With Items')
"""

