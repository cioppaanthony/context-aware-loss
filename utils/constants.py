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
from utils.argument_parser import args

# Variable Definition
TRAIN_INDEX = 0
VALID_INDEX = 1
TEST_INDEX = 2

# List of the games per set
LIST_TRAIN = "/listgame_Train_300.npy"
LIST_VALID = "/listgame_Valid_100.npy"
LIST_TEST = "/listgame_Test_100.npy"

# Feature and labels types
FEATURE_TYPE = "ResNET_PCA512.npy"
LABEL_NAME = "/Labels.json"

# Events as annotated in SoccerNet
EVENT_DICTIONARY = {"soccer-ball": 0, "soccer-ball-own": 0, "r-card": 1, "y-card": 1, "yr-card": 1,
                                 "substitution-in": 2}

# K Matrix from the arguments
K_MATRIX = np.array([[args.K1G,args.K1C,args.K1S],[args.K2G,args.K2C,args.K2S],[args.K3G,args.K3C,args.K3S],[args.K4G,args.K4C,args.K4S]])*args.framerate