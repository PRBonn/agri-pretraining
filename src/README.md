# Self Supervised Agriculture Network

Refers to <a href='https://github.com/facebookresearch/barlowtwins'>Barlow Twins (FAIR)</a> for the original PyTorch implementation of the baseline and to <a href='https://gitlab.ipb.uni-bonn.de/gianmarco.roggiolani/SSLPR'>Baseline Evaluation</a> for the experiments on the pre-trained model using PyTorch. 

## Setup

The `install.py` script provides a virtual environment (conda), install the requirements and create a package which allows us to import modules and functions from anywhere in the system like every other package: 

``` py
    import self_supervised_agrinet.stuff
```

You don't have to reinstall anything after changing your code. 


## Directory structure

``` bash
└── lightning_project
    ├── project_name
    │   ├── experiments
    │   ├── config
    │   │   └── config.yaml
    │   ├── datasets
    │   │   ├── datasets.py
    │   │   └── __init__.py
    │   ├── models
    │   │   ├── blocks.py
    │   │   ├── __init__.py
    │   │   ├── loss.py
    │   │   └── models.py
    │   ├── test.py
    │   ├── train.py
    │   └── utils
    │       └── __init__.py
    ├── README.md
    ├── requirements.txt
    ├── install.py
    └── setup.py
```

## How to use 

1. Personalize the config.yaml 
    * Write the correct path to your dataset 
    * Define the characteristics of your training procedure - i.e. max number of epochs, learning rates and batch size
    * Define your model - i.e. backbone, dropout, projector network and embedding size 

2. If your dataset has a different structure, re-write the datalaoder 

3. In utils.py you can switch on/off the different augmentations or change their probabilities 

4. Run the code just as ```python train.py``` 

5. Your weights will be saved in the experiments folder, together with the log file for tensorboard (to visualize the loss)
