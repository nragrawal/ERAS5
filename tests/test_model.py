import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from model.train import SimpleCNN, train, show_augmented_images
from torchvision import transforms
from model.augmentation_utils import get_transforms, show_augmented_images

def test_model_architecture():
    model = SimpleCNN()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape should be (1, 10)"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"

def test_model_training():
    accuracy, model = train()
    assert accuracy >= 95.0, f"Model accuracy {accuracy:.2f}% is below 95%"

def test_model_input():
    model = SimpleCNN()
    
    # Test various batch sizes
    batch_sizes = [1, 4, 16]
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, 1, 28, 28)
        try:
            output = model(test_input)
            assert output.shape == (batch_size, 10), f"Failed for batch size {batch_size}"
        except Exception as e:
            pytest.fail(f"Model failed to process batch size {batch_size}: {str(e)}")

def test_model_feature_dimensions():
    """Test the intermediate feature dimensions in the model"""
    model = SimpleCNN()
    x = torch.randn(1, 1, 28, 28)
    
    # Test first conv + pool
    x1 = model.pool(torch.relu(model.conv1(x)))
    assert x1.shape == (1, 10, 13, 13), "First conv+pool output shape incorrect"
    
    # Test second conv + pool
    x2 = model.pool(torch.relu(model.conv2(x1)))
    assert x2.shape == (1, 10, 5, 5), "Second conv+pool output shape incorrect"
    
    # Test flattened dimension
    x3 = x2.view(-1, 10 * 5 * 5)
    assert x3.shape == (1, 250), "Flattened dimension incorrect"

def test_model_activation_ranges():
    """Test if model outputs are in the expected range"""
    model = SimpleCNN()
    test_input = torch.randn(10, 1, 28, 28)
    output = model(test_input)
    
    # Test if outputs are in reasonable range for softmax
    assert torch.all(output >= -100) and torch.all(output <= 100), \
        "Model outputs are in unexpected range"
    
    # Test if different classes are being predicted
    predictions = torch.argmax(output, dim=1)
    unique_predictions = torch.unique(predictions)
    assert len(unique_predictions) > 1, \
        "Model is predicting only one class"

def test_model_augmentation():
    """Test the image augmentation functionality"""
    # Create a dummy image
    original = torch.ones(1, 28, 28)
    
    # Convert to PIL
    from torchvision.transforms import ToPILImage, ToTensor
    pil_image = ToPILImage()(original)
    
    # Apply augmentation
    aug_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
    ])
    
    # Convert back to tensor
    augmented = ToTensor()(aug_transform(pil_image))
    
    # Test that augmentation changed the image
    assert not torch.allclose(original, augmented), \
        "Augmentation did not modify the image"
    
    # Test that augmentation preserved image size
    assert original.shape == augmented.shape, \
        "Augmentation changed image dimensions"