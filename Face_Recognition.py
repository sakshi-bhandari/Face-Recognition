
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
from inception_blocks_v2 import *

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# Face Verification

# Following command creates the model for face images.

FRmodel = faceRecoModel(input_shape=(3, 96, 96))

print("Total Params:", FRmodel.count_params())

def triplet_loss(y_true, y_pred, alpha = 0.2):
    
    # Implementation of the triplet loss as defined by formula 
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2] 
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(positive-anchor), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(negative-anchor), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist-neg_dist+alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    
    return loss

# Loading the pre-trained model - FaceNet is trained by minimizing the triplet loss. 

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

database = {}
database["abc"] = img_to_encoding("images/abc.png", FRmodel)

def verify(image_path, identity, database, model):
    
    # Function that verifies if the person on the "image_path" image is "identity".
    
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above.
    encoding = img_to_encoding(image_path, model)
    
    # Step 2: Compute distance with identity's image
    dist = np.linalg.norm(encoding-database[identity])
    
    # Step 3: Open the door if dist < 0.7, else don't open
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome in!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False
      
    return dist, door_open

# following code verifies whether two images are similar
# "images/camera_0.jpg" represents a random image, which user has to add to the required folder
verify("images/camera_0.jpg", "abc", database, FRmodel)

# Face Recognition

def who_is_it(image_path, database, model):
    
    # Implements face recognition for the office by finding who is the person on the image_path image.
    
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above.
    encoding = img_to_encoding(image_path, model)
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value
    min_dist = 200
    
    # Loop over the database dictionary's names and encodings.
    for name in database:
        
        # Compute L2 distance between the target "encoding" and the current db_enc from the database.
        dist = np.linalg.norm(encoding-database[name])

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name.
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity

who_is_it("images/camera_0.jpg", database, FRmodel)
