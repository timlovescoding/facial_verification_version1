# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:49:55 2019

@author: Tim
"""


size = 96 # Size of images you want
  
from PIL import Image
import os

# Alter the path to your directory where the images are at:
# Have the non-_____your image___ sets and the image you want to classify separated by files

path = "/Users/Tim/Documents/Python/Face_verification/resizing_images/new2/"
dirs = os.listdir( path )

# Remember to resize for both set of files (your image and non-image)

for item in dirs:
   if os.path.isfile(path+item):
       im = Image.open(path+item)
       imResize = im.resize((size,size), Image.ANTIALIAS)
       output_file_name = os.path.join(path, "resized_" + item)
       imResize.save(output_file_name , 'JPEG', quality= 95)

print('All done!')