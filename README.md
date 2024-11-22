# MNIST CNN Classifier

A simple CNN-based classifier for the MNIST dataset with automated testing and CI/CD pipeline.

## Project Overview

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The model is designed to be lightweight (<25,000 parameters) while achieving >94% accuracy in a single epoch of training.

## Model Architecture

The CNN architecture consists of:
- 3 convolutional layers with ReLU activation and batch normalization
- Max pooling layers after each convolution
- Dropout for regularization
- Progressive channel width (16->24->16)
- 2 fully connected layers (48 units, 10 outputs)
- Total parameters: ~14,642

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