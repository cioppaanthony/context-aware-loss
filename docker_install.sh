#!/bin/bash
# Copyright (c) 2020 Anthony Cioppa, Adrien Deliège and Silvio Giancola

set -euf -o pipefail

echo --------------------------------
echo APT-GET
echo --------------------------------
apt-get -y update
apt-get -y upgrade
apt-get -y install htop

echo --------------------------------
echo APT-GET PYTHON
echo --------------------------------
apt-get -y install python-pip
apt-get -y install python3-pip python3-dev
apt-get -y install python3-tk

echo --------------------------------
echo PIP INSTALL
echo --------------------------------
python3 -m pip install --upgrade pip
python3 -m pip install numpy==1.17.3
python3 -m pip install tabulate==0.8.5
python3 -m pip install tqdm==4.23.1
python3 -m pip install h5py==2.10.0
python3 -m pip install matplotlib==3.0.3
python3 -m pip install opencv-python-headless==4.1.2.30
python3 -m pip install opencv-contrib-python-headless==4.1.2.30
python3 -m pip install tensorflow==2.0.0
