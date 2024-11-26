# MNIST CNN Classifier

A simple CNN-based classifier for the MNIST dataset with automated testing and CI/CD pipeline.

## Project Overview

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The model is designed to be lightweight (<25,000 parameters) while achieving >95% accuracy in a single epoch of training.

## Model Architecture

The CNN architecture consists of:
- 2 convolutional layers with ReLU activation (10 channels each)
- Max pooling layers after each convolution
- 2 fully connected layers (16 units, 10 outputs)
- Total parameters: 5,196
  - Conv1: 100 parameters (3×3×1×10 + 10)
  - Conv2: 910 parameters (3×3×10×10 + 10)
  - FC1: 4,016 parameters (250×16 + 16)
  - FC2: 170 parameters (16×10 + 10)

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