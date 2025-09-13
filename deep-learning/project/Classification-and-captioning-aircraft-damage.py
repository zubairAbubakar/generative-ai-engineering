
# Introduction¶
# In this project, you will classify aircraft damage using a pre-trained VGG16 model and generate captions using a Transformer-based pretrained model.
#
# Project Overview
# Aircraft damage detection is essential for maintaining the safety and longevity of aircraft. Traditional manual inspection methods are time-consuming and prone to human error. This project aims to automate the classification of aircraft damage into two categories: "dent" and "crack." For this, we will utilize feature extraction with a pre-trained VGG16 model to classify the damage from aircraft images. Additionally, we will use a pre-trained Transformer model to generate captions and summaries for the images.
#
# Aim of the Project
# The goal of this project is to develop an automated model that accurately classifies aircraft damage from images. By the end of the project, you will have trained and evaluated a model that utilizes feature extraction from VGG16 for damage classification. This model will be applicable in real-world damage detection within the aviation industry. Furthermore, the project will showcase how we can use a Transformer-based model to caption and summarize images, providing a detailed description of the damage.
#
# Final Output
# A trained model capable of classifying aircraft images into "dent" and "crack" categories, enabling automated aircraft damage detection.
# A Transformer-based model that generates captions and summaries of images

# To achieve the above objectives, you will complete the following tasks:
#
# Task 1: Create a valid_generator using the valid_datagen object
# Task 2: Create a test_generator using the test_datagen object
# Task 3: Load the VGG16 model
# Task 4: Compile the model
# Task 5: Train the model
# Task 6: Plot accuracy curves for training and validation sets
# Task 7: Visualizing the results
# Task 8: Implement a Helper Function to Use the Custom Keras Layer
# Task 9: Generate a caption for an image using the using BLIP pretrained model
# Task 10: Generate a summary of an image using BLIP pretrained model

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import zipfile
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.applications import VGG16
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
import random

