#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:25:23 2019

@author: sameer
"""

import exiftool
import pprint
import os
import json

path = './geo'
list1 = []
list1 = os.listdir(path)
list2 = []
for fname in list1:
    path2 = os.path.join(path,fname)

    with exiftool.ExifTool() as et:
        metadata= et.get_metadata(path2)
        print(metadata['Composite:GPSPosition'])
        
    break
list2.sort()
with open('exif.json', 'w') as outfile:  
    json.dump(list2,outfile, indent=4, sort_keys = True)