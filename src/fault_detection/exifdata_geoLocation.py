#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 19:31:23 2019

@author: sameer
"""


import exiftool
import pprint
import os
import json



try: 
        with exiftool.ExifTool() as et:
            metadata= et.get_metadata('4_Thermal_Panorama_Count.jpg')
            print(metadata['Composite:GPSPosition'])
    
except:
    print('No geo location')