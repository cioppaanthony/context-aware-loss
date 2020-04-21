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

from utils.argument_parser import args
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
from tqdm import tqdm
import cv2
import os
np.seterr(divide='ignore', invalid='ignore')

def NMS(detections, delta):

	# Array to put the results of the NMS
	detections_tmp = np.copy(detections)
	detections_NMS = np.zeros(detections.shape)-1

	# Loop over all classes
	for i in np.arange(detections.shape[-1]):
		# Stopping condition
		while(np.max(detections_tmp[:,i]) >= 0):

			# Get the max remaining index and value
			max_value = np.max(detections_tmp[:,i])
			max_index = np.argmax(detections_tmp[:,i])

			detections_NMS[max_index,i] = max_value

			detections_tmp[int(np.maximum(-(delta/2)+max_index,0)): int(np.minimum(max_index+int(delta/2), detections.shape[0])) ,i] = -1

	return detections_NMS

def graphVideo(data, labels, network, savepath, NMS_on=True):


	# Definition of some variables
	start = 0
	last = False
	timestamps = None
	chunk_size = args.chunksize*args.framerate
	receptive_field = int(args.receptivefield*args.framerate/2)

	# Arrays for saving the segmentations, the detections and the detections scores
	prediction = np.zeros((data.shape[0],labels.shape[1]))
	timestamps_scores = np.zeros((data.shape[0],labels.shape[1]))-1

	# Labels one hot encoded
	labels_one_hot = np.where(labels == 0, 1, 0)

	# Preprocessing of the feature data
	data_expanded = np.expand_dims(data, axis=0)
	if data_expanded.shape[-1] != 3:
		data_expanded = np.expand_dims(data_expanded, axis=-1)


	# Loop over the entire game, chunk by chunk
	while True:

		# ----------------------------------------
		# Predict the result for the current chunk
		# ----------------------------------------

		# Get the output for that chunk and retrive the segmentations and the detections
		tmp_output = network.predict_on_batch(data_expanded[:,start:start+chunk_size])
		output = tmp_output[0][0]
		timestamps = tmp_output[1][0]

		# Store the detections confidence score for the chunck
		timestamps_long_score = np.zeros(output.shape)-1

		# Store the detections and confidence scores in a chunck size array
		for i in np.arange(timestamps.shape[0]):
			timestamps_long_score[ int(np.floor(timestamps[i,1]*(output.shape[0]-1))) , int(np.argmax(timestamps[i,2:])) ] = timestamps[i,0]
		
		# ------------------------------------------
		# Store the result of the chunk in the video
		# ------------------------------------------

		# For the first chunk
		if start == 0:
			prediction[0:chunk_size-receptive_field] = output[0:chunk_size-receptive_field]
			timestamps_scores[0:chunk_size-receptive_field] = timestamps_long_score[0:chunk_size-receptive_field]
		
		# For the last chunk
		elif last:
			prediction[start+receptive_field:start+chunk_size] = output[receptive_field:]
			timestamps_scores[start+receptive_field:start+chunk_size] = timestamps_long_score[receptive_field:]
			break
		
		# For every other chunk
		else:
			prediction[start+receptive_field:start+chunk_size-receptive_field] = output[receptive_field:chunk_size-receptive_field]
			timestamps_scores[start+receptive_field:start+chunk_size-receptive_field] = timestamps_long_score[receptive_field:chunk_size-receptive_field]
		
		# ---------------
		# Loop Management
		# ---------------
		
		start += chunk_size - 2 * receptive_field
		if start + chunk_size >= data.shape[0]:
			start = data.shape[0] - chunk_size 
			last = True

	# Apply Non Maxima Suppression if required
	if NMS_on:
		timestamps_scores = NMS(timestamps_scores,2*receptive_field)

	# Get all the detections whose confidence is over 0.34
	timestamps_final = np.where(timestamps_scores >= 0.34, 1.35, 0)

	np.save(savepath[:-4] + "_segmentations.npy", 1-prediction)
	np.save(savepath[:-4] + "_labels.npy", labels_one_hot)
	np.save(savepath[:-4] + "_detections.npy", timestamps_final)
	np.save(savepath[:-4] + "_detections_scores.npy", timestamps_scores)

def confusion_matrix_single_game(targets, detections, delta, threshold):

	# Get all targets indexes for each class
	num_classes = targets.shape[1]

	TP = list()
	FP = list()
	FN = list()

	# Iterate over all classes
	for i in np.arange(num_classes):
		gt_indexes = np.where(targets[:,i]==1)[0]
		pred_indexes = np.where(detections[:,i] >=threshold)[0]
		pred_scores = detections[pred_indexes,i]

		# If there are no groundtruths
		if len(gt_indexes) == 0:
			TP.append(0)
			FP.append(len(pred_indexes))
			FN.append(0)
			continue

		# If there are no predictions
		if len(pred_indexes) == 0:
			TP.append(0)
			FP.append(0)
			FN.append(len(gt_indexes))
			continue

		# Iterate over all groundtruths
		TP_class = 0
		FP_class = 0
		FN_class = 0
		remove_indexes = list()

		for gt_index in gt_indexes:
			# Get the predictions which are within the delta interval of each 
			max_score = -1
			max_index = None
			for pred_index, pred_score in zip(pred_indexes, pred_scores):
				# The two indexes are very close to each other, choose the one with the greatest score
				if abs(pred_index-gt_index) <= delta/2 and pred_score > max_score and pred_index not in remove_indexes:
					max_score = pred_score
					max_index = pred_index
			# If, for this groundtruth, no predictions could fit
			if max_index is None:
				FN_class += 1
			# If there is one good prediction
			else:
				TP_class += 1
				remove_indexes.append(max_index)
		
		FP_class = len(pred_indexes)-len(remove_indexes)

		TP.append(TP_class)
		FP.append(FP_class)
		FN.append(FN_class)

	return TP, FP, FN


def compute_confusion_matrix(targets, detections, delta, threshold):

	TP = np.array([0]*targets[0].shape[1])
	FP = np.array([0]*targets[0].shape[1])
	FN = np.array([0]*targets[0].shape[1])

	for target, detection in zip(targets, detections):
		TP_tmp, FP_tmp, FN_tmp = confusion_matrix_single_game(target, detection, delta, threshold)
		TP += np.array(TP_tmp)
		FP += np.array(FP_tmp)
		FN += np.array(FN_tmp)

	return TP, FP, FN

def compute_precision_recall_curve(targets, detections, delta, NMS_on):

	# 200 confidence thresholds between [0,1]
	thresholds = np.linspace(0,1,200)

	# Store the precision and recall points
	precision = list()
	recall = list()

	# Apply Non-Maxima Suppression if required
	detections_NMS = list()
	if NMS_on:
		for detection in detections:
			detections_NMS.append(NMS(detection,delta))
	else:
		detections_NMS = detections

	# Get the precision and recall for each confidence threshold
	for threshold in thresholds:
		TP, FP, FN = compute_confusion_matrix(targets, detections_NMS, delta, threshold)
		p = np.nan_to_num(TP/(TP+FP))
		r = np.nan_to_num(TP/(TP+FN))

		precision.append(p)
		recall.append(r)
	precision = np.array(precision)
	recall = np.array(recall)

	# Sort the points based on the recall, class per class
	for i in np.arange(precision.shape[1]):
		index_sort = np.argsort(recall[:,i])
		precision[:,i] = precision[index_sort,i]
		recall[:,i] = recall[index_sort,i]

	return precision, recall

def compute_mAP(precision, recall):

	# Array for storing the AP per class
	AP = np.array([0.0]*precision.shape[-1])

	# Loop for all classes
	for i in np.arange(precision.shape[-1]):

		# 11 point interpolation
		for j in np.arange(11)/10:

			index_recall = np.where(recall[:,i] >= j)[0]

			possible_value_precision = precision[index_recall,i]
			max_value_precision = 0

			if possible_value_precision.shape[0] != 0:
				max_value_precision = np.max(possible_value_precision)

			AP[i] += max_value_precision

	mAP_per_class = AP/11

	return np.mean(mAP_per_class)

def delta_curve(targets, detections, savepath, NMS_on):

	mAP = list()

	for delta in tqdm((np.arange(12)*5 + 5)*args.framerate):

		precision, recall = compute_precision_recall_curve(targets, detections, delta, NMS_on)

		mAP.append(compute_mAP(precision, recall))

	return mAP

def average_mAP(dataset, network, savepath=None, NMS_on=True):

	# Get the features and labels from the set
	features = dataset.features
	labels = dataset.labels
	
	# Compute the chunk size and receptive field from the arguùments
	chunk_size = args.chunksize*args.framerate
	receptive_field = int(args.receptivefield*args.framerate/2)

	# Get the correct stamps for all the data
	detections = list()
	targets = list()


	# Loop over all of the data and labels
	for data, label in tqdm(zip(features, labels)):

		# Set up controle variables
		last = False
		start = 0

		# Set up the array for storing the final results
		timestamps_final = np.zeros((data.shape[0],label.shape[1]))-1

		# Preparing the labels and sending them to the list
		labels_one_hot = np.where(label == 0, 1, 0)
		targets.append(labels_one_hot)

		# Expand the data in the last dimension if needed
		data_expanded = np.copy(data)
		if data.shape[-1] != 3:
			data_expanded = np.expand_dims(data_expanded, axis=-1)

		# Loop over the entire match
		while True:

			# ----------------------------------------
			# Predict the result for the current chunk
			# ----------------------------------------

			# Prepare the batch made of one chunk for the network
			input_data = np.expand_dims(data_expanded[start:start+chunk_size],axis=0)

			# Send the batch into the network and retrieve the results
			tmp_output = network.predict_on_batch(input_data)
			output = tmp_output[0][0]
			timestamps = tmp_output[1][0]

			# Expanding the timestamps so they are in the same shape as the labels
			timestamps_long = np.zeros(output.shape)-1

			# Put each score in the timestamp to the correct place
			for i in np.arange(timestamps.shape[0]):
				timestamps_long[ int(np.floor(timestamps[i,1]*(output.shape[0]-1))) , int(np.argmax(timestamps[i,2:])) ] = timestamps[i,0]

			# ------------------------------------------
			# Store the result of the chunk in the video
			# ------------------------------------------

			# For the first chunk
			if start == 0:
				timestamps_final[0:chunk_size-receptive_field] = timestamps_long[0:chunk_size-receptive_field]

			# For the last chunk
			elif last:
				timestamps_final[start+receptive_field:start+chunk_size] = timestamps_long[receptive_field:]
				break

			# For every other chunk
			else:
				timestamps_final[start+receptive_field:start+chunk_size-receptive_field] = timestamps_long[receptive_field:chunk_size-receptive_field]
			
			# ---------------
			# Loop Management
			# ---------------

			# Update the index
			start += chunk_size - 2 * receptive_field
			# Check if we are at the last index of the game
			if start + chunk_size >= data.shape[0]:
				start = data.shape[0] - chunk_size 
				last = True
		
		# Append the results of the predictions to the list
		detections.append(timestamps_final)

	# Once all predictions and labels are retrieved, compute the mAP for each curve
	mAP = delta_curve(targets, detections, savepath, NMS_on)

	# Compute the average mAP
	integral = 0.0
	for i in np.arange(len(mAP)-1):
		integral += 5*(mAP[i]+mAP[i+1])/2
	a_mAP = integral/(5*(len(mAP)-1))

	if savepath is not None:
		myfile = open(savepath, 'a')
		myfile.write(str(a_mAP)+";")
		myfile.close()
	return a_mAP