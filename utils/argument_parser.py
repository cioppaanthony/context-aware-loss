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

import argparse

parser = argparse.ArgumentParser()

# -------------------------------
# Arguments concerning the inputs
# -------------------------------
parser.add_argument('--datasetpath', '-d', help="Path to the SoccerNet dataset. Either the original SoccerNet, or the preprocessed features provided", type=str, required=True)
parser.add_argument('--savepath', '-s', help="Path to save the results of the training or the inference", type=str, required=True)
parser.add_argument('--featuretype', '-t', help="Type of the features to use", type=str, default="ResNET_PCA512.npy")
parser.add_argument('--framerate', '-f', help="Framerate of the input features, SoccerNet features are by default at 2 fps", type=int, default=2)
parser.add_argument('--weights', '-w', help="Network weights to use for inference", type=str, default=None)

# ---------------------------------
# Arguments concerning the training
# ---------------------------------
parser.add_argument("--epochs", '-e', help="Number of epochs on which to train the network", type=int, default=1000)
parser.add_argument("--epochsvalidation", '-ev', help="Evaluate the network on the validation set once every x epochs", type=int, default=20)
parser.add_argument("--learningrate", '-lr', help="Learning rate for the training", type=float, default=0.001)
parser.add_argument("--lambdacoord", '-lc', help="Weight of the coordinates of the event in the detection loss", type=float, default=5.0)
parser.add_argument("--lambdanoobj", '-ln', help="Weight of the no object detection in the detection loss", type=float, default=0.5)
parser.add_argument("--lossweightsegmentation", '-lws', help="Weight of the segmentation loss compared to the detection loss", type=float, default=0.002)
parser.add_argument("--lossweightdetection", '-lwd', help="Weight of the detection loss", type=float, default=1.0)


# ---------------------------------------------
# Arguments concerning the network architecture
# ---------------------------------------------
parser.add_argument("--chunksize", '-cs', help="Size of the chunks to feed the network (in seconds)", type=int, default=120)
parser.add_argument("--receptivefield", '-rf', help="Temporal receptive field of the network (in seconds)", type=int, default=40)
parser.add_argument("--capsules", '-c', help="Number of capsules in the segmentation module", type=int, default=16)
parser.add_argument("--numbertimestamps", '-nts', help= "Maximum number of detection per chunk for the action spotting", type=int, default=5)

# -----------------------------------
# Arguments concerning the evaluation
# -----------------------------------
parser.add_argument("--NMSON", help="Non-maxima suppression is applied by default, use --NMSON 0 not to use the NMS", type = int, default=1)
parser.add_argument("--indexvideo", help="Index of the video (unused for now)", type = int, default=1)

# ----------------------------------------------------------------------------------
# Arguments concerning the parameters of the Segmentation loss (Time Shift Encoding)
# ----------------------------------------------------------------------------------
parser.add_argument("--K1G", '-k1g', help="parameter of the K matrix n 1 (in seconds) for Goals", type=int, default=-20)
parser.add_argument("--K2G", '-k2g', help="parameter of the K matrix n 2 (in seconds) for Goals", type=int, default=-10)
parser.add_argument("--K3G", '-k3g', help="parameter of the K matrix n 3 (in seconds) for Goals", type=int, default=60)
parser.add_argument("--K4G", '-k4g', help="parameter of the K matrix n 4 (in seconds) for Goals", type=int, default=90)
parser.add_argument("--K1C", '-k1c', help="parameter of the K matrix n 1 (in seconds) for Cards", type=int, default=-20)
parser.add_argument("--K2C", '-k2c', help="parameter of the K matrix n 2 (in seconds) for Cards", type=int, default=-10)
parser.add_argument("--K3C", '-k3c', help="parameter of the K matrix n 3 (in seconds) for Cards", type=int, default=10)
parser.add_argument("--K4C", '-k4c', help="parameter of the K matrix n 4 (in seconds) for Cards", type=int, default=20)
parser.add_argument("--K1S", '-k1s', help="parameter of the K matrix n 1 (in seconds) for Substitutions", type=int, default=-40)
parser.add_argument("--K2S", '-k2s', help="parameter of the K matrix n 2 (in seconds) for Substitutions", type=int, default=-20)
parser.add_argument("--K3S", '-k3s', help="parameter of the K matrix n 3 (in seconds) for Substitutions", type=int, default=10)
parser.add_argument("--K4S", '-k4s', help="parameter of the K matrix n 4 (in seconds) for Substitutions", type=int, default=20)
# If m * sqrt(c) <= 1: -> Sphere is totally included in the feature space
parser.add_argument("--mplus", '-mp', help="m plus hit zone", type=float, default=0.1)
parser.add_argument("--mminus", '-mm', help="m minus miss zone", type=float, default=0.9)

args = parser.parse_args()
