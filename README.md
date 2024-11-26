# MNIST CNN Classifier

A simple CNN-based classifier for the MNIST dataset with automated testing and CI/CD pipeline.

## Project Overview

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The model is designed to be lightweight (<25,000 parameters) while achieving >95% accuracy in a single epoch of training.

## Model Architecture

The CNN architecture consists of:
- 2 convolutional layers with ReLU activation (10 channels each)
- Max pooling layers after each convolution (2x2)
- 2 fully connected layers (64 units, 10 outputs)
- Total parameters: 17,724
  - Conv1: 100 parameters (3×3×1×10 + 10)
  - Conv2: 910 parameters (3×3×10×10 + 10)
  - FC1: 16,064 parameters (250×64 + 64)
  - FC2: 650 parameters (64×10 + 10)

## Features

- Data augmentation with random rotation
- ASCII art visualization of augmented images
- Modular code structure with separate augmentation utilities
- Comprehensive test suite
- CI/CD pipeline with GitHub Actions

## Project Structure

.
├── model/
│   ├── __init__.py          # Package initialization
│   ├── train.py             # Main training script and CNN model
│   └── augmentation_utils.py # Image augmentation and visualization
├── tests/
│   └