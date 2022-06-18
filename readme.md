# Online shopping assistant

## About

The objective of this project is distinguish whether 
a mug has its logo presented in front of the camera.

## Installation
IDE: Pycharm

Python version: 3.8

Python environment: venv

Framework: Tensorflow

Camera call: OpenCV

## Usage
Run the main.py function using Pycharm configuration.
This function will use

## Dataset
The dataset used to train the model is part of 
"A Fast Data Collection and Augmentation Procedure for Object Recognition"
from [http://ai.stanford.edu/](http://ai.stanford.edu/~asaxena/robotdatacollection/).
Some python script in folder *dataCollectionTool* are used to collect and sort data.

The objective of this dataset is for object detection with annotation,
part of which contains a subset for mugs.

The author classified those mugs with logos and without logos into two folders for this training task.

## Model
The model used in for this task 
is from the Tensorflow tutorial, 
as it's a simple binary classification task.

The tutorial link: [Click here](https://www.tensorflow.org/tutorials/images/classification)

## Saving Model 
The pretrained model is saved in *checkpoints* folder

## Demo
A video demo is here: *src/output.avi*
The demo is based on the saved model: *checkpoints/my_model_1.h5*

