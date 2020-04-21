"""
----------------------------------------------------------------------------------------
Copyright (c) 2020 - see AUTHORS file

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
----------------------------------------------------------------------------------------
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Conv2D, Reshape, Lambda, Concatenate, MaxPooling2D
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
from utils.argument_parser import args
K.set_image_data_format('channels_last')


def baseline(input_shape, dim_capsule, num_classes, receptive_field, nb_detections):
	
	# -----------------------------------
	# Feature input (chunks of the video)
	# -----------------------------------

	main_input = Input(shape=input_shape, dtype='float32', name='main_inp')

	# -------------------------------------
	# Temporal Convolutional neural network
	# -------------------------------------

	# Base Convolutional Layers
	conv1 = Conv2D(input_shape[1]//4, (1,input_shape[1]), strides = (1,1), padding = 'valid', activation = 'relu')(main_input)
	conv2 = Conv2D(input_shape[1]//16, (1,1), strides = (1,1), padding = 'valid', activation = 'relu')(conv1)
	
	# Temporal Pyramidal Module
	conv3_1 = Conv2D(8,(int(np.ceil(receptive_field/7)),1), strides = (1,1), padding = 'same', activation = 'relu')(conv2)
	conv3_2 = Conv2D(16,(int(np.ceil(receptive_field/3)),1), strides = (1,1), padding = 'same', activation = 'relu')(conv2)
	conv3_3 = Conv2D(32,(int(np.ceil(receptive_field/2)),1), strides = (1,1), padding = 'same', activation = 'relu')(conv2)
	conv3_4 = Conv2D(64,(int(np.ceil(receptive_field)),1), strides = (1,1), padding = 'same', activation = 'relu')(conv2)
	concat = Concatenate(axis=-1)([conv2,conv3_1,conv3_2,conv3_3,conv3_4])

	# Capsule preparation
	conv4 = Conv2D(dim_capsule*num_classes,(3,1),strides=(1,1),padding='same')(concat) 
	resh = Reshape((input_shape[0],dim_capsule,num_classes))(conv4)
	resh = BatchNormalization(axis=1, fused=False)(resh)
	capsules = Activation('sigmoid')(resh)

	# -------------------
	# Segmentation output
	# -------------------
	output1 = Lambda(lambda x: K.sqrt(K.sum(K.square(x-0.5),axis=2,keepdims=False)*4/dim_capsule))(capsules)
	


	# ---------------
	# Spotting module
	# ---------------

	# Concatenation of the segmentation score to the capsules as input to the detection module
	resh_out = Lambda(lambda x: 1-x)(output1)
	resh_out = Reshape((input_shape[0],1,num_classes))(resh_out)
	conv4_conc = Concatenate()([conv4, resh_out])

	# Convolutional CNN for the detection
	conv4_relu = Activation('relu')(conv4_conc)
	conv4_relu = MaxPooling2D((3,1),strides=(2,1))(conv4_relu) 
	conv5 = Conv2D(32,(3,1), strides=(1,1), padding = 'same', activation='relu')(conv4_relu)
	conv5_max = MaxPooling2D((3,1),strides=(2,1))(conv5) 
	conv6 = Conv2D(16,(3,1), strides=(1,1), padding = 'same', activation='relu')(conv5_max)
	conv6_max = MaxPooling2D((3,1),strides=(2,1))(conv6) 
	conv6_resh = Reshape((1,1,int(np.prod(conv6_max.shape[1:]))))(conv6_max)

	# Activation on the confidence score timestamp of the detection
	conv7 = Conv2D(nb_detections*2,(1,1), strides=(1,1), padding = 'same')(conv6_resh)
	resh_conv7 = Reshape((nb_detections,2))(conv7)
	resh_conv7 = Activation('sigmoid')(resh_conv7)

	# Activation on one-hot encoded classes
	conv8 = Conv2D(nb_detections*num_classes,(1,1), strides=(1,1), padding = 'same')(conv6_resh)
	resh_conv8 = Reshape((nb_detections,num_classes))(conv8)
	resh_conv8 = Activation('softmax')(resh_conv8)

	# ----------------
	# Detection output
	# ----------------
	output2 = Concatenate()([resh_conv7, resh_conv8])
	
	# ---------------------
	# Creation of the model
	# ---------------------
	model = Model(main_input, [output1, output2])
	
	return model
