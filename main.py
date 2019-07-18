
"""
Purpose: Facial Verification System Version 1 (Naive Implementation)

@author: Tim

*huge credits to the set-up done by the CNN course on coursera by Prof Andrew Ng
*huge credits to the paper written by FaceNet and CNN weights from OpenFace


"""
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from block import *
import matplotlib.pyplot as plt


FRmodel = faceRecoModel(input_shape=(3, 96, 96))  # 1) Face images was reshaped into 96x96 images
load_weights_from_FaceNet(FRmodel)  # 2) Loading pre-trained weights of the inception network


# 3) Setting up the database, getting the encoding of face images from different people
# Note: Just change the path to where your images are at.
database = {}
database["tim"] = img_to_encoding("images/resized_tim.jpg", FRmodel)
database["ly"]  = img_to_encoding("images/resized_ly.jpg",FRmodel)
database["ben"]=  img_to_encoding("images/resized_ben.jpg",FRmodel)
database["younes"]= img_to_encoding("images/resized_younes.jpg",FRmodel)
database["bertrand"] = img_to_encoding("images/resized_bertrand.jpg",FRmodel)
database["jeremy"] = img_to_encoding("images/resized_jeremy.jpg",FRmodel)
database["jenice"] = img_to_encoding("images/resized_jenice.jpg",FRmodel)
database["jiaren"] = img_to_encoding("images/resized_jiaren.jpg",FRmodel)

# 4) Verify the "photo taken" and see whehter is it the identity claimed.
def verify( camera_picture , identity , database, model):

    # Finding the minimum distance between encoding to verify the person based off the camera picture.
    encoding = img_to_encoding( camera_picture , model)  # Encoding based off the camera picture taken
    minimum  = 100 #initialising (just be a high number, it's temporary)

    for person, _ in database.items():
        
         #dist = np.linalg.norm(encoding - database[person]) # Compute the distance between the encoding and database
         dist = np.sum((encoding - database[person])**2)
         if dist < minimum:
            minimum  = dist  # Re-assigned new minimum
            confirmed_identity = person # the identity of the person based off the camera picture
    
    if minimum > 0.65:  # If the distance is over a threshold, it is not close enough to be anyone in the database
        confirmed_identity = "a stranger"
        
    if confirmed_identity == identity:
        
        fig, (ax1, ax2) = plt.subplots(1,2)
        
        ax1.imshow(image_identity_plot)
        ax1.set_title(" You claim to be " + str(identity))
       
        ax2.imshow(image_camera_plot)
        ax2.set_title("Photo taken: Welcome Back" + " " + str(identity) + "!")
       
    
    else:
        
        fig , (ax1, ax2) = plt.subplots(1,2)
        
        ax1.imshow(image_identity_plot)
        ax1.set_title(" You claim to be " + str(identity))
        
        ax2.imshow(image_camera_plot)
        ax2.set_title("Access denied, you are  " + str(confirmed_identity) + "!" )
        
        
    return minimum



# Ask who is the person. ( GET THE IDENTITY )
identity = input("Who is this?\n"
                 "1. tim\n2. ly\n3. ben \n4. younes \n5. bertrand \n"
                 "6. jeremy\n7. jenice \n8. jiaren \n Type it:")

# Plot the Identity Photo database! (Plot once you tell the identity)
identity_picture =  ("images/resized_"+identity+".jpg")
image_identity = cv2.imread ("C:/Users/Tim/Documents/Python/Face_verification/" + identity_picture )
image_identity_plot = cv2.cvtColor(image_identity, cv2.COLOR_BGR2RGB) # OpenCV is BGR, matplot lib is RGB



# Now get another picture from the camera ( HYPOTHETICALLY TAKING A PHOTO NOW)

picture_shown = input("Camera photo taken is: ( HYPOTHETICALLY TAKING A PHOTO NOW)\n"
                 "1. tim\n2. ly\n3. ben \n4. younes \n5. bertrand \n"
                 "6. jeremy\n7. jenice \n8. jiaren \n9. stranger: ken\n Type it:")

camera_picture   = ("images/resized_"+picture_shown+"_camera.jpg")

# Plot the Camera Photo! (in the verify function)
image_camera = cv2.imread ("C:/Users/Tim/Documents/Python/Face_verification/" + camera_picture )
image_camera_plot = cv2.cvtColor(image_camera, cv2.COLOR_BGR2RGB) # OpenCV is BGR, matplot lib is RGB


# Search through the database and determine whether is it the person he/she claims to be!
distance_check = verify( camera_picture , identity , database, FRmodel)

print('Finish')