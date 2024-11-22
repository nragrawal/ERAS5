# MNIST CNN Classifier

A simple CNN-based classifier for the MNIST dataset with automated testing and CI/CD pipeline.

## Project Overview

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The model is designed to be lightweight (<25,000 parameters) while achieving >94% accuracy in a single epoch of training.

## Model Architecture

The CNN architecture consists of:
- 2 convolutional layers with ReLU activation (10 channels each)
- Max pooling layers after each convolution
- Dropout (0.2) for regularization
- 2 fully connected layers (64 units, 10 outputs)
- Total parameters: 17,724
  - Conv1: 100 parameters
  - Conv2: 910 parameters
  - FC1: 16,064 parameters
  - FC2: 650 parameters

## Requirements

- Python 3.12
- numpy 1.26.4
- pillow 10.2.0
- PyTorch 2.2.0
- torchvision 0.17.0
- pytest 8.0.0

## Project Structure

.
├── model/
│   └ 