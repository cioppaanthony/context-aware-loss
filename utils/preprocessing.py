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
import utils.constants as C
from utils import argument_parser
import random

args = argument_parser.args

def labelToCategorical(labels, sequence_length_first_half, sequence_length_second_half, framerate=2, num_classes=3):
	"""
	Transforms the labels from the json format to a numpy array with size (num_class, num_frames)
	Every element is set to zero except for the frame at which an event occurs where it is set to 1
	"""

	labels_first_half = np.zeros((sequence_length_first_half, num_classes))
	labels_second_half = np.zeros(( sequence_length_second_half, num_classes))

	for annotation in labels["annotations"]:
		time = annotation["gameTime"]
		event = annotation["label"]

		half = int(time[0])

		minutes = int(time[-5:-3])
		seconds = int(time[-2::])

		frame = framerate * ( seconds + 60 * minutes ) 

		if event not in C.EVENT_DICTIONARY:
			continue
		label = C.EVENT_DICTIONARY[event]

		if half == 1:
			labels_first_half[frame][label] = 1

		if half == 2:
			labels_second_half[frame][label] = 1
		

	return labels_first_half, labels_second_half


def rulesToCombineShifts(shift_from_last_event, shift_until_next_event, params):
	
	s1  = shift_from_last_event
	s2  = shift_until_next_event
	K = params
	
	if s1 < K[2]:
		value = s1
	elif s1 < K[3]:
		if s2 <= K[0]:
			value = s1
		else:
			if (s1-K[2])/(K[3]-K[2]) < (K[1]-s2)/(K[1]-K[0]):
				value = s1
			else:
				value = s2
	else:
		value = s2
		
	return value


def oneHotToShifts(onehot, params):
	
	nb_frames = onehot.shape[0]
	nb_actions = onehot.shape[1]
	
	Shifts = np.empty(onehot.shape)
	
	for i in range(nb_actions):
		
		x = onehot[:,i]
		K = params[:,i]
		shifts = np.empty(nb_frames)
		
		loc_events = np.where(x == 1)[0]
		nb_events = len(loc_events)
		
		if nb_events == 0:
			shifts = np.full(nb_frames, K[0])
		elif nb_events == 1:
			shifts = np.arange(nb_frames) - loc_events
		else:
			loc_events = np.concatenate(([-K[3]],loc_events,[nb_frames-K[0]]))
			for j in range(nb_frames):
				shift_from_last_event = j - loc_events[np.where(j >= loc_events)[0][-1]]
				shift_until_next_event = j - loc_events[np.where(j < loc_events)[0][0]]
				shifts[j] = rulesToCombineShifts(shift_from_last_event, shift_until_next_event, K)
		
		Shifts[:,i] = shifts
	
	return Shifts


def getChunks(features, labels):

	# get indexes of labels
	indexes=list()
	for i in np.arange(labels.shape[1]):
		indexes.append(np.where(labels[:,i] == 0)[0].tolist())

	# Positive chunks
	positives_chunks_features = list()
	positives_chunks_labels = list()

	chunk_size = args.chunksize*args.framerate
	receptive_field = args.receptivefield*args.framerate

	for event in indexes:
		for element in event:
			shift = random.randint(-chunk_size+receptive_field, -receptive_field)
			start = element + shift
			if start < 0:
				start = 0
			if start+chunk_size >= features.shape[0]:
				start = features.shape[0]-chunk_size-1
			positives_chunks_features.append(features[start:start+chunk_size])
			positives_chunks_labels.append(labels[start:start+chunk_size])


	# Negative chunks
	number_of_negative_chunks = np.floor(len(positives_chunks_labels)/labels.shape[1])+1
	negatives_chunks_features = list()
	negatives_chunks_labels = list()

	negative_indexes = getNegativeIndexes(labels, C.K_MATRIX)

	counter = 0
	while counter < number_of_negative_chunks and counter < len(negative_indexes):
		selection = random.randint(0, len(negative_indexes)-1)
		start = random.randint(negative_indexes[selection][0], negative_indexes[selection][1]-chunk_size)
		if start < 0:
			start = 0
		if start+chunk_size >= features.shape[0]:
			start = features.shape[0]-chunk_size-1
		negatives_chunks_features.append(features[start:start+chunk_size])
		negatives_chunks_labels.append(labels[start:start+chunk_size])
		counter += 1

	positives_array_features = np.array(positives_chunks_features)
	positives_array_labels = np.array(positives_chunks_labels)
	negatives_array_features = np.array(negatives_chunks_features)
	negatives_array_labels = np.array(negatives_chunks_labels)

	inputs = None
	targets = None

	if positives_array_features.shape[0] > 0 and negatives_array_features.shape[0] > 0:
		inputs = np.copy(np.concatenate((positives_array_features, negatives_array_features), axis=0))
		targets = np.copy(np.concatenate((positives_array_labels, negatives_array_labels), axis=0))
	elif negatives_array_features.shape[0] == 0:
		inputs = np.copy(positives_array_features)
		targets = np.copy(positives_array_labels)
	else:
		inputs = np.copy(negatives_array_features)
		targets = np.copy(negatives_array_labels)
	if positives_array_features.shape[0] == 0 and negatives_array_features.shape[0] == 0:
		print("No chunks could be retrieved...")
	
	
	# Put loss to zero outside receptive field
	targets[:,0:int(np.ceil(receptive_field/2)),:] = -1
	targets[:,-int(np.ceil(receptive_field/2)):,:] = -1

	return inputs, targets

def getTimestampTargets(labels):

	targets = np.zeros((labels.shape[0],args.numbertimestamps,2+labels.shape[-1]), dtype='float')

	for i in np.arange(labels.shape[0]):

		time_indexes, class_values = np.where(labels[i]==0)

		counter = 0

		for time_index, class_value in zip(time_indexes, class_values):

			# Confidence
			targets[i,counter,0] = 1.0 
			# frame index normalized
			targets[i,counter,1] = time_index/(args.chunksize*args.framerate)
			# The class one hot encoded
			targets[i,counter,2+class_value] = 1.0
			counter += 1

			if counter >= args.numbertimestamps:
				print("More timestamp than what was fixed... A lot happened in that chunk")
				break

	return targets




def getNegativeIndexes(labels, params):

	zero_one_labels = np.zeros(labels.shape)
	for i in np.arange(labels.shape[1]):
		zero_one_labels[:,i] = 1-np.logical_or(np.where(labels[:,i] >= params[3,i], 1,0),np.where(labels[:,i] <= params[0,i], 1,0))
	zero_one = np.where(np.sum(zero_one_labels, axis=1)>0, 0, 1)

	zero_one_pad = np.append(np.append([1-zero_one[0],], zero_one, axis=0), [1-zero_one[-1]], axis=0)
	zero_one_pad_shift = np.append(zero_one_pad[1:], zero_one_pad[-1])

	zero_one_sub = zero_one_pad - zero_one_pad_shift

	zero_to_one_index = np.where(zero_one_sub == -1)[0]
	one_to_zero_index = np.where(zero_one_sub == 1)[0]


	if zero_to_one_index[0] > one_to_zero_index[0]:
		one_to_zero_index = one_to_zero_index[1:]
	if zero_to_one_index.shape[0] > one_to_zero_index.shape[0]:
		zero_to_one_index = zero_to_one_index[:-1]

	list_indexes = list()

	for i,j in zip(zero_to_one_index, one_to_zero_index):
		if j-i >= args.chunksize*args.framerate:
			list_indexes.append([i,j])

	return list_indexes