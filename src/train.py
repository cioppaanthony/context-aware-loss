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

import time
import numpy as np
from tqdm import tqdm
import losses
import model
import utils.preprocessing
import utils.constants as C
from utils import io_module
from utils import argument_parser
from utils import evaluation
import tensorflow.keras
from tensorflow.keras import backend as K


if __name__ == "__main__":

	# ------------------
	# Load the arguments
	# ------------------

	args = argument_parser.args
	savepath = args.savepath
	NMS_on = (args.NMSON==1)

	# -----------------
	# Load the datasets 
	# -----------------

	print("Creating datasets")
	trainset = io_module.Dataset(args.datasetpath, C.LIST_TRAIN, args.featuretype, framerate=args.framerate, num_classes=3)
	validationset = io_module.Dataset(args.datasetpath, C.LIST_VALID, args.featuretype, framerate=args.framerate, num_classes=3)

	trainset.storeFeatures()
	validationset.storeFeatures()

	# -----------------
	# Training settings
	# -----------------

	# Load the network
	network = model.baseline(trainset.input_shape, args.capsules, trainset.num_classes, args.receptivefield*args.framerate, args.numbertimestamps)
	
	# Load the margins
	hit = np.array([args.mplus]*trainset.num_classes)
	miss = np.array([args.mminus]*trainset.num_classes)
	
	# Load the segmentation and detection losses
	model_STL = losses.SegmentationTSELoss(params = C.K_MATRIX, hit_radius = hit, miss_radius = miss)
	model_DSL = losses.DetectionSpottingLoss(lambda_coord=args.lambdacoord, lambda_noobj=args.lambdanoobj)

	# Load the optimizer
	optimizer = tensorflow.keras.optimizers.Adam(lr=args.learningrate, beta_1=0.9, beta_2=0.999, amsgrad=False)

	# Compile the network with the losses, the optimizers and loss weights
	network.compile(loss=[model_STL, model_DSL], optimizer=optimizer, loss_weights=[args.lossweightsegmentation, args.lossweightdetection])
	print(network.summary())

	# -------------
	# TRAINING PART
	# -------------

	# For saving the best results
	best_mAP = 0.0

	# Epoch loop
	for epoch in np.arange(args.epochs):

		# Some variables for the segmentation and detection metrics
		metric_train_segmentation = 0.0
		metric_train_detection = 0.0

		# Learning rate decay
		current_learning_rate = K.get_value(network.optimizer.lr)
		new_learning_rate = current_learning_rate - (args.learningrate-(10**-6))/args.epochs
		K.set_value(network.optimizer.lr,new_learning_rate)

		# Iteration over all games videos of the dataset
		for data, label in tqdm(zip(trainset.features, trainset.labels)):

			# Get random chunks from the video
			# Number of chunks depends on the number of events
			# Then build the segmentation and detection targets
			inputs, segmentation_targets = utils.preprocessing.getChunks(data,label)
			detection_targets = utils.preprocessing.getTimestampTargets(segmentation_targets)
			inputs = np.expand_dims(inputs, axis=-1)

			# Train it on that batch
			metric_train = network.train_on_batch(inputs, [segmentation_targets, detection_targets] )
			
			# Update the segmentation and detection loss metrics
			metric_train_segmentation += metric_train[1]
			metric_train_detection += metric_train[2]

		# Display the metrics for this epoch
		print("\nEPOCH N ", str(epoch), "Mean training metric for segmentation per game: ",metric_train_segmentation/trainset.labels.shape[0])
		print("EPOCH N ", str(epoch), "Mean training metric for detection per game: ",metric_train_detection/trainset.labels.shape[0], '\n')
		
		# Save the latest network weights
		network.save_weights(savepath + "/last_weights.h5")

		# Once every 20 epochs, compute the average-mAP and save the network weight if it is the best current one
		if epoch % args.epochsvalidation == 0 and epoch != 0:
			current_mAP = evaluation.average_mAP(validationset, network, savepath+"/Validation_Average_mAP.log", NMS_on = NMS_on)
			print("Average-mAP: ", current_mAP)
			if current_mAP > best_mAP:
				# Save the network with the best weights
				network.save_weights(savepath+"/best_average_mAP.h5")
				best_mAP = current_mAP
				print("New best average-mAP of ", best_mAP, " at epoch ", epoch)


		