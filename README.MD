# A Context-Aware Loss Function for Action Spotting in Soccer Videos

This repository contains the code to reproduce the main results of the paper: "A Context-Aware Loss Function for Action Spotting in Soccer Videos". The paper is accepted and will be presented in June 2020 at the CVPR Conference. The preprint can be found on arXiv following this link: [Context-Aware Loss Function](https://arxiv.org/abs/1912.01326).

In this paper, we design a novel loss that leverages the temporal context around an action spot. We heavily penalize the frames far-distant from the action and steadily decrease the penalty for the frames gradually closer to the action. The frames just before the action are not penalized to avoid providing misleading information as its occurrence is uncertain. However, those just after the action are heavily penalized as we know for sure that the action has occurred. This loss is illustrated in the following figure.

<p align="center"><img src="img/Abstract.png" width="480"></p>

Our work is focused on action spotting in soccer videos. We use the SoccerNet dataset [openaccess.thecvf.com](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w34/Giancola_SoccerNet_A_Scalable_CVPR_2018_paper.pdf) which is a large scale dataset comprising 500 soccer videos. There are three categories of actions to spot in this dataset: Goals, Cards and Substitutions.

This repository provides all training and testing functions necessary to reproduce the main results of the paper. It also provides a notebook to reproduce two of the figures of the paper: Figure 6 for the three classes (goal, card, substitution) and Figures 7 for different games.

here is a short video summary of our method.
[INSERT VIDEO HERE]


## Getting Started (under construction)

The following instructions will help you install the required libraries and the dataset to run the code. The code runs in <code>python 3</code> and was tested inside a nvidia-docker with the folowing base image: <code>tensorflow:18.02-py3</code> which can be found following this link: [NVIDIA](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow/tags).

### Prerequisites

Wether you are using the docker image or not, here are the versions of the libraries that are used:

```
numpy==1.17.2
tabulate==0.8.5
tqdm==4.23.1
h5py==2.10.0
matplotlib==3.0.3
opencv-python-headless==4.1.2.30
opencv-contrib-python-headless==4.1.2.30
tensorflow==2.0.0
```

### Installing on docker with the provided install bash file

If you are using the nvidia-docker, follow these steps:

```
1. Create the docker.
2. Clone the repository.
3. cd into the repository.
4. Type the following command line to install all libraries: "bask docker_install.sh".
```

At this step, all the required libraries are installed.

### Installing with pip

To install all required libraries, run the following command lines (might require <code>sudo</code>).

```
python3 -m pip install numpy==1.17.2
python3 -m pip install tabulate==0.8.5
python3 -m pip install tqdm==4.23.1
python3 -m pip install h5py==2.10.0
python3 -m pip install matplotlib==3.0.3
python3 -m pip install opencv-python-headless==4.1.2.30
python3 -m pip install opencv-contrib-python-headless==4.1.2.30
python3 -m pip install tensorflow==2.0.0
```
At this step, all the required libraries are installed.

### Installing the SoccerNet Dataset


For the installation of the SoccerNet dataset, please follow the instructions on the original author's github: [SoccerNet](https://github.com/SilvioGiancola/SoccerNet-code)

## Training the network (under construction)

The code for training the network is located inside the <code>src</code> folder under the name <code>train.py</code>.

Before running the code, create a folder to save the network weights and the metric saved at each epoch. Then, simply go to the <code>src</code> folder and run the following command:

```
python3 train.py -d path/to/soccernet/data/ -s path/to/save/trained/weights/ -t ResNET_PCA512.npy
```

Feel free to explore the different parameters that are available for this command in the <code>utils/argument_parser.py</code> file. If you wish to use another feature, simply change the <code>-t</code> parameter with whatever feature type available in the SoccerNet dataset.

## Inference on the testset (under construction)

The code for testing the network is located inside the <code>src</code> folder under the name <code>test.py</code>.

Again, before running this code, create a folder for saving the results. We will save multiple results including the average-mAP, which is the metric used to compare the algorithms in SoccerNet, some numpy arrays for each half time of each game in the testset that contain the detections which will be used to reproduce the results of Figure 7 and some other numpy arrays that contain counting metrics which will be used to reproduce the results of Figure 6. Once you created the folder, simply run the following command:

```
python3 test.py -d path/to/soccernet/data/ -s path/to/save/results/ -t ResNET_PCA512.npy -n path/to/trained/weights/
```

Note that if you do not want to train the network, we provide some weights that we trained ourself as described in the paper in the <code>weights</code> folder, under the name <code>soccernet_trained_weights.h5</code>

You can now move on to the jupyter notebook in the <code>notebooks</code> folder and run it.
```
1. cd into the notebooks directory.
2. Type in the terminal: "jupyter notebook" and go to the "graphs" file.
3. Replace the path in the first block by the fodler where you saved your test results: /path/to/save/results/.
4. Run the whole notebook.
```

If you used our pre-trained weights for the test, you should get something similar to the following graphs:

<p align="center"><img src="img/figure_7.png" width="480"></p>
<p align="center"><img src="img/figure_6.png" width="480"></p>

## Authors

* **Anthony Cioppa** - Code contribution: *train - test - evaluation - notebooks - utils*
* **Adrien Deliège** - Code contribution: *Model - losses*
* **Silvio Giancola** - Code contribution: *evaluation - [SoccerNet](https://github.com/SilvioGiancola)*

## License (under construction)

## Acknowledgments

* A. Cioppa is funded by the FRIA, Belgium.
* This work is supported by the DeepSport project of the Walloon region, Belgium, and by the King Abdullah University of Science and Technology (KAUST) Office of Sponsored Research. 
