# Contextual Non-Stationary Network Selection Multi-Armed Bandit

## Overview

This is a repository of a project about a Contextual Multi-Armed Bandit for Wireless Network selection. This project is supported by ROUTE56 and Universitat Politècnica de Catalunya. This project has had a paper published in IEEE Globecom 2023. It can be read [here](https://ieeexplore.ieee.org/document/10437363).

This repository is divided into 2 parts: simmulation and results representation. The simmulation code is the most interesting one. It its the part of code in charge of running the whole simmulation. On the other hand, the results code is in charge of creating a visual representation of the results obtained from the simmulation code. The simmulation code has been developed using python3.9 and the representation code uses matlab.

    
## Install and set-up

In order to install this repository first clone it into your computer (for example, ussing SSH keys):
    
    git clone git@github.com:lekesen/Network-Selection-MAB.git

In order to be able to execute code, firstly we need to create some folders.
    
    cd Network-Selection-MAB/
    mkdir results

In order to execute the simmulation code, some python libraries will need to be installed. In order to keep clean working environments, we recommend using python virtual environments.

    
### Creating a python virtual environment

In order to create a python virtual environment, we will use the virtualenv package:
    
    sudo apt-get update
    sudo apt-get install python3-virtualenv

Once we have downloaded the tool for creating virtual environments, go to the folder where you want to create the virtual environment and run:
    
    virtualenv --python='python3.x' venv
    
This command will create a python virtual environment with version 3.x. For this repository we recommend using python 3.9.7.

To "activate" the virtual environment:
    
    source venv/bin/activate
    
And to "deactivate" it:
    
    deactivate

    
### Downloading python dependencies

Inside your `virtualenv` and inside the Network-Selection-MAB folder call:
    
    cd simmulation_code
    pip install -r requirements.txt

This command will install all the dependencies defined in requirements.txt. In case an error occurs during installation, first try running:
    
    sudo apt install python3-dev
    
    
### Downloading other necessary dependencies

In order to work with `tensorflow`, it is usually recommended to use a GPU. This usually tends to be a very tedious process because of installation problems, so a brief installation guide will be explained in the following lines. The official guide provided by tensorflow may be read [here](https://www.tensorflow.org/install/gpu). This guide is only for Nvidia GPU's. For running the simmulations, you need to install:

* **NVIDIA GPU driver**: In our case we used an NVIDIA GTX 960 with driver version 470.86. You can install the drivers for your GPU in [here](https://www.nvidia.com/download/index.aspx?lang=en-us).
* **CUDA Toolkit**: in our case we used NVIDIA Toolkit 11.4.3 (for Ubuntu 20.04). The list of downloadable versions may be found [here](https://developer.nvidia.com/cuda-toolkit-archive).
* **cuDNN**: you should install `cudNN library` and `cuDNN Runtime Library`. In our case we used cuDNN v8.2.4 for CUDA 11.4 (and Ubuntu 20.04). The list of downloadable versions may be found [here](https://developer.nvidia.com/rdp/cudnn-archive) and an in-detail installation guide [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) (You really should read it!).

Once you have downloaded all of this dependencies, now you should check if tensorflow is able to detect the gpu. In order to do this activate the `virtualenv` and run:
    
    python
    import tensorflow as tf
    tf.config.list_physical_devices('GPU')

This last command should return some 'Information' logs, but it should also return that at least 1 GPU has been detected.
    
## How to use

In order to execute the simmulation (and inside `virtualenv` and `Network-Selection-MAB/simmulation_code`):
    
    python main.py

If you only want to execute the simmulation for a few algorithms, you can select which ones you want in `main.py`.
    
In order to view results, open matlab and go to the folder `Network-Selection-MAB/representation_code` and execute:

* **load_results.m**: for observing total regret.
* **load_ba.m**: for observing best-action percentage at each time-stemp.
* **load_time.m**: observe average execution time.
   
