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

from utils import io_module
from utils import argument_parser
from tqdm import tqdm
import time
import numpy as np
import utils.constants as C
import utils.preprocessing
import losses
import model
import tensorflow.keras
from utils import evaluation
from tensorflow.keras import backend as K


if __name__ == "__main__":
	
	# ------------------
	# Load the arguments
	# ------------------

	args = argument_parser.args
	NMS_on = (args.NMSON==1)

	# ----------------
	# Load the testset
	# ----------------

	testset = io_module.Dataset(args.datasetpath, C.LIST_TEST, args.featuretype, framerate=args.framerate, num_classes=3)
	testset.storeFeatures()
	
	# ----------------
	# Testing settings
	# ----------------

	# Load the network and its weights
	network = model.baseline(testset.input_shape, args.capsules, testset.num_classes, args.receptivefield*args.framerate, args.numbertimestamps)
	network.load_weights(args.weights)
	
	# Evaluate and display the average-mAP
	average_mAP = evaluation.average_mAP(testset, network, args.savepath+"/Test_Average_mAP.log", NMS_on = NMS_on)
	print("Average-mAP: ", average_mAP)

	# For the all videos of the testset, get the graphs and numpy arrays of the results
	counter = 0
	for data, label in tqdm(zip(testset.features, testset.labels)):

		evaluation.graphVideo(data,label, network, args.savepath+"/Test_"+str(counter)+".png", NMS_on = NMS_on)
		counter += 1