# Signature Verification Using Siamese Network

This notebook demonstrates how to implement signature verification using a Siamese Network. A Siamese Network is a type of neural network architecture that uses two identical subnetworks, each taking one of the two input signatures. The networks are joined at their outputs by a similarity function. The objective is to learn a similarity metric between the outputs of the two subnetworks. This metric can then be used to determine if the two input signatures belong to the same person. This approach is particularly useful for tasks like signature verification where the number of samples per class is limited.

## Table of Contents

1.  [Import Libraries](#import-libraries)
2.  [Setup](#setup)
3.  [Helper Functions](#helper-functions)
    *   [Loading and Preprocessing Images](#loading-and-preprocessing-images)
    *   [Visualizing Samples](#visualizing-samples)
    *   [Creating Image Pairs](#creating-image-pairs)
    *   [Building the Base Network](#building-the-base-network)
    *   [Euclidean Distance Layer](#euclidean-distance-layer)
    *   [Building the Siamese Network](#building-the-siamese-network)
    *   [Training the Model](#training-the-model)
    *   [Plotting Training History](#plotting-training-history)
    *   [Signature Verification](#signature-verification)
4.  [Signature Verification System](#signature-verification-system)
    *   [Load Data](#load-data)
    *   [Visualize Data](#visualize-data)
    *   [Create Pairs](#create-pairs)
    *   [Train Model](#train-model)
    *   [Evaluate and Plot Results](#evaluate-and-plot-results)
    *   [Save Model](#save-model)
    *   [Perform Verification](#perform-verification)

## Import Libraries

Essential libraries are imported for data manipulation, image processing, deep learning model building (TensorFlow/Keras), plotting, and Google Drive integration.

## Setup

Sets random seeds for reproducibility and mounts Google Drive to access the dataset.

## Helper Functions

### Loading and Preprocessing Images

The `load_images` function reads signature images from specified 'genuine' and 'forged' subfolders. It resizes images to 128x128, converts them to grayscale, and normalizes pixel values to the range [0, 1]. It also extracts a person ID from the filename and assigns a label (1 for genuine, 0 for forged).

### Visualizing Samples

The `visualize_samples` function displays a few sample genuine and forged signatures to provide an overview of the data.

### Creating Image Pairs

The `create_pairs` function generates positive and negative image pairs from the loaded dataset. Positive pairs consist of two signatures from the same person, while negative pairs consist of two signatures from different people. The number of positive and negative pairs is balanced.

### Building the Base Network

The `build_base_network` function defines the convolutional neural network (CNN) that will be used as the shared subnetwork in the Siamese architecture. It consists of convolutional layers, max-pooling layers, and dense layers to extract feature embeddings from input images.

### Euclidean Distance Layer

The `euclidean_distance` function calculates the Euclidean distance between the feature vectors produced by the two branches of the Siamese network. This distance serves as a measure of similarity between the two input images.

### Building the Siamese Network

The `build_siamese_network` function constructs the complete Siamese network. It takes two input images, passes each through the shared base network, computes the Euclidean distance between their feature vectors, and finally uses a dense layer with a sigmoid activation to predict the probability that the two signatures belong to the same person.

### Training the Model

The `train_model` function handles the training process. It splits the generated image pairs into training and testing sets, separates the left and right images for the network inputs, compiles the Siamese model with an appropriate optimizer and loss function, and trains it using the training data. It also evaluates the model on the test set.

### Plotting Training History

The `plot_training` function visualizes the training and validation loss and accuracy over epochs, allowing for monitoring of the model's performance during training.

### Signature Verification

The `verify_signatures` function takes a trained model and paths to two image files. It loads, preprocesses, and feeds these images to the model to get a similarity score. Based on a threshold, it determines if the signatures are from the same person and visualizes the images used for verification.

## Signature Verification System

This section orchestrates the entire signature verification process.

### Load Data

Loads images, labels, and person IDs from the specified dataset path using `load_images`.

### Visualize Data

Displays sample genuine and forged signatures using `visualize_samples`.

### Create Pairs

Generates image pairs for training the Siamese network using `create_pairs`.

### Train Model

Trains the Siamese network using the prepared pairs and labels via `train_model`.

### Evaluate and Plot Results

Evaluates the trained model on the test set and plots the training history using `plot_training`.

### Save Model

Saves the trained model to an HDF5 file for later use.

### Perform Verification

Demonstrates how to use the trained model to verify if two given signatures (`/content/sig1.PNG` and `/content/sig2.png`, and `/content/sig_1.png` and `/content/sig_2.png` in the examples) are from the same person using the `verify_signatures` function.
