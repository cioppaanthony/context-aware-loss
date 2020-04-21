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

from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
from utils.argument_parser import args
tf.compat.v1.disable_eager_execution()

# ------------------------------------------
# Segmentation with Time Shift Encoding loss
# ------------------------------------------
def ComputeSegmentationTSELoss(y_true, y_pred, params, hit_radius, miss_radius):
	loss = K.sum(-K.minimum(0.,K.sign(y_true))*K.maximum(0.,-K.log(y_pred+(1.-y_pred)*(K.minimum(K.maximum(y_true,params[0,:]),params[1,:])-params[0,:])/(params[1,:]-params[0,:]))+np.log(miss_radius))+(1.-K.maximum(0.,K.sign(-y_true)))*K.maximum(0.,K.maximum(-K.log(1.-y_pred+K.abs(y_true)/params[2,:])+np.log(1.-hit_radius),-K.log(y_pred+(K.minimum(y_true,params[3,:])-params[3,:])/(params[2,:]-params[3,:]))+np.log(miss_radius))))
	return loss

# ---------------------------------------------
# Segmentation with Shift Encoding loss wrapper
# ---------------------------------------------
def SegmentationTSELoss(params, hit_radius, miss_radius):
	def SegmentationTSE(y_true, y_pred):
		return ComputeSegmentationTSELoss(y_true, y_pred, params, hit_radius, miss_radius)
	return SegmentationTSE

# --------------
# Detection loss
# --------------
def ComputeDetectionSpottingLoss(y_true, y_pred, lambda_coord, lambda_noobj):
	y_pred = permute_ypred_for_matching(y_true,y_pred)
	loss = K.sum(  y_true[:,:,0]*lambda_coord*K.square(y_true[:,:,1]-y_pred[:,:,1])  +  y_true[:,:,0]*K.square(y_true[:,:,0]-y_pred[:,:,0]) +  (1-y_true[:,:,0])*lambda_noobj*K.square(y_true[:,:,0]-y_pred[:,:,0]) +  y_true[:,:,0]*K.sum(K.square(y_true[:,:,2:]-y_pred[:,:,2:]),axis=-1)) #-y_true[:,:,0]*K.sum(y_true[:,:,2:]*K.log(y_pred[:,:,2:]),axis=-1)
	return loss

# ----------------------
# Detection loss wrapper
# ----------------------
def DetectionSpottingLoss(lambda_coord, lambda_noobj):
	def DetectionSpotting(y_true, y_pred):
		return ComputeDetectionSpottingLoss(y_true, y_pred, lambda_coord, lambda_noobj)
	return DetectionSpotting



# -----------------------------
# Iterative one-to-one matching
# -----------------------------
def permute_ypred_for_matching(y_true, y_pred):

	# Permutation for the predictions and targets 
	# Applied before the detection loss is computed

	alpha = y_true[:,:,0]
	x = y_true[:,:,1]
	p = y_pred[:,:,1]
	nb_pred = args.numbertimestamps
	
	D = K.abs(tf.tile(K.expand_dims(x,axis=-1),(1,1,nb_pred)) - tf.tile(K.expand_dims(p,axis=-2),(1,nb_pred,1)))
	D1 = 1-D
	Permut = 0*D
	
	alpha_filter = tf.tile(K.expand_dims(alpha,-1),(1,1,nb_pred))
	
	v_filter = alpha_filter
	h_filter = 0*v_filter + 1 
	D2 = v_filter * D1

	for i in range(nb_pred):
		D2 = v_filter * D2
		D2 = h_filter * D2
		A = tf.one_hot(K.argmax(D2,axis=-1),nb_pred)
		B = v_filter * A * D2
		C = tf.transpose(tf.one_hot(K.argmax(B,axis=-2),nb_pred), perm=[0, 2, 1])
		E = v_filter * A * C
		Permut = Permut + E
		v_filter = (1-K.sum(Permut,axis=-1))*alpha
		v_filter = tf.tile(K.expand_dims(v_filter,-1),(1,1,nb_pred))
		h_filter = 1-K.sum(Permut, axis=-2)
		h_filter = tf.tile(K.expand_dims(h_filter,-2),(1,nb_pred,1))
	
	v_filter = 1-alpha_filter
	D2 = v_filter * D1
	D2 = h_filter * D2
	
	for i in range(nb_pred):
		D2 = v_filter * D2
		D2 = h_filter * D2
		A = tf.one_hot(K.argmax(D2,axis=-1),nb_pred)
		B = v_filter * A * D2
		C = tf.transpose(tf.one_hot(K.argmax(B,axis=-2),nb_pred), perm=[0, 2, 1])
		E = v_filter * A * C
		Permut = Permut + E
		v_filter = (1-K.sum(Permut,axis=-1))*(1-alpha)
		v_filter = tf.tile(K.expand_dims(v_filter,-1),(1,1,nb_pred))
		h_filter = 1-K.sum(Permut, axis=-2)
		h_filter = tf.tile(K.expand_dims(h_filter,-2),(1,nb_pred,1))
	
	permutation = K.argmax(Permut,axis=-1)
	permuted = tf.gather(y_pred,permutation,batch_dims=1)
		
	return permuted