import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from model.train import SimpleCNN, train

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
    assert accuracy > 95.0, f"Model accuracy {accuracy:.2f}% is below 95%"

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