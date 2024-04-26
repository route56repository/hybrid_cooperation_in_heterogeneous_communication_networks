# Non-Stationary Network Selection Multi-Armed Bandit

## Overview

This is a repository of a project about a Multi-Armed Bandit for Wireless Network selection. This project is supported by ROUTE56 and Universitat Politècnica de Catalunya. This project has had a paper published in IEEE Globecom 2021. It can be read [here](https://ieeexplore.ieee.org/document/9681963).

This repository is divided into 2 parts: simmulation and results representation. The simmulation code is the most interesting one. It its the part of code in charge of running the whole simmulation. On the other hand, the results code is in charge of creating a visual representation of the results obtained from the simmulation code. The simmulation code has been developed using python3.9 and the representation code uses matlab.

    
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
    
    
## How to use

In order to execute the simmulation (and inside `virtualenv` and `/simmulation_code`):
    
    python main.py

If you only want to execute the simmulation for a few algorithms, you can select which ones you want in `main.py`.
    
In order to view results, open matlab and go to the folder `/representation_code` and execute:

* **load_results.m**: for observing total regret.
* **load_ba.m**: for observing best-action percentage at each time-stemp.
* **load_time.m**: observe average execution time.
   
