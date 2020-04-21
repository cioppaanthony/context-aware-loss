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

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

def load_predictions(path):

	path = path+"/Test_"
	targets = list()
	detections = list()
	segmentations = list()

	for i in np.arange(200):
		targets.append(np.load(path+str(i)+"_labels.npy"))
		detections.append(np.load(path+str(i)+"_detections_scores.npy"))
		segmentations.append(np.load(path + str(i) + "_segmentations.npy"))

	return targets, detections, segmentations

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


def countEvents(targets, detections, NMS_on=True):

	num_classes = targets[0].shape[1]

	best_thresholds = list()

	for delta in tqdm((np.arange(12)*5)+5):

		best_thresholds.append(0.0)
		best_F1 = 0.0

		detections_NMS = list()
		if NMS_on:
			for detection in detections:
				detections_NMS.append(NMS(detection,delta))
		else:
			detections_NMS = detections

		for threshold in np.linspace(0,1,200):

			TP = np.array([0]*num_classes)
			FP = np.array([0]*num_classes)
			FN = np.array([0]*num_classes)

			for target, detection in zip(targets, detections_NMS):

				TP_tmp, FP_tmp, FN_tmp = confusion_matrix_single_game(target, detection, delta, threshold)
				TP += np.array(TP_tmp)
				FP += np.array(FP_tmp)
				FN += np.array(FN_tmp)

			F1 = np.mean(2*TP/(2*TP+FP+FN))
			if F1 > best_F1:
				best_F1 = F1
				best_thresholds[-1] = threshold

	#print("Best thresholds: ", best_thresholds)


	TP_delta = list()
	FP_delta = list()
	FN_delta = list()
	

	for delta, threshold in tqdm(zip((np.arange(12)*5)+5, best_thresholds)):


		TP = np.array([0]*num_classes)
		FP = np.array([0]*num_classes)
		FN = np.array([0]*num_classes)
		

		detections_NMS = list()
		if NMS_on:
			for detection in detections:
				detections_NMS.append(NMS(detection,delta))
		else:
			detections_NMS = detections

		for target, detection in zip(targets, detections_NMS):

			TP_tmp, FP_tmp, FN_tmp = confusion_matrix_single_game(target, detection, delta, threshold)
			TP += np.array(TP_tmp)
			FP += np.array(FP_tmp)
			FN += np.array(FN_tmp)
			

		TP_delta.append(TP)
		FP_delta.append(FP)
		FN_delta.append(FN)

	TP_delta = np.array(TP_delta)
	FP_delta = np.array(FP_delta)
	FN_delta = np.array(FN_delta)

	return TP_delta, FP_delta, FN_delta
