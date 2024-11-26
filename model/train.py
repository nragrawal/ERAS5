import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
import os
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First conv: Input(1, 28, 28) -> Conv(10, 26, 26) -> Pool(10, 13, 13)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        
        # Second conv: Input(10, 13, 13) -> Conv(10, 11, 11) -> Pool(10, 5, 5)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3)
        
        # Pooling layer used after each conv
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # Input: Flattened output from conv2 (10 * 5 * 5 = 250 features)
        self.fc1 = nn.Linear(10 * 5 * 5, 64)  # 250 -> 32
        # Output: 10 classes for digits 0-9
        self.fc2 = nn.Linear(64, 10)  # 32 -> 10
        
    def forward(self, x):
        # First conv block: 28x28 -> 26x26 -> 13x13
        x = self.pool(torch.relu(self.conv1(x)))
        
        # Second conv block: 13x13 -> 11x11 -> 5x5
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Flatten: 10 channels * 5 * 5 pixels = 250 features
        x = x.view(-1, 10 * 5 * 5)
        
        # Fully connected layers with ReLU activation
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def show_augmented_images(originals, augmenteds, title="Augmentation Examples"):
    """Display images as ASCII art in console"""
    print(f"\n{title}")
    print("-" * 80)
    
    for idx in range(3):
        # Convert tensors to numpy arrays and normalize to 0-1
        orig_img = originals[idx].squeeze().numpy()
        aug_img = augmenteds[idx].squeeze().numpy()
        
        # Resize to smaller dimensions for ASCII art
        small_orig = (orig_img[::2, ::2] > 0.5).astype(int)
        small_aug = (aug_img[::2, ::2] > 0.5).astype(int)
        
        # Convert to ASCII
        ascii_chars = [' ', '░', '▒', '▓', '█']
        
        print(f"\nPair {idx + 1}:")
        print("Original:")
        for row in small_orig:
            print(''.join(ascii_chars[int(val * 4)] for val in row))
        
        print("\nAugmented:")
        for row in small_aug:
            print(''.join(ascii_chars[int(val * 4)] for val in row))
        
        print("-" * 80)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define augmentation transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(10),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Basic transform for visualization
    basic_transform = transforms.ToTensor()
    
    # Load dataset
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Show augmentation examples
    test_dataset = datasets.MNIST('data', train=True, download=True, transform=basic_transform)
    
    # Prepare lists for original and augmented images
    original_images = []
    augmented_images = []
    
    # Get 3 sample images
    for i in range(3):
        original_image = test_dataset[i][0]
        original_images.append(original_image)
        
        # Apply augmentation
        aug_transform = transforms.Compose([
            transforms.RandomRotation(10),
        ])
        
        # Convert to PIL image for transforms
        pil_image = transforms.ToPILImage()(original_image)
        augmented_image = basic_transform(aug_transform(pil_image))
        augmented_images.append(augmented_image)
    
    # Show all images in console
    show_augmented_images(original_images, augmented_images, "3 Augmentation Examples")
    
    # Initialize model, loss function, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    # Training loop for one epoch
    model.train()
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}')
    
    accuracy = 100. * correct / total
    print(f'Training Accuracy: {accuracy:.2f}%')
    
    # Save model with timestamp for versioning
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), f'models/mnist_model_{timestamp}.pth')
    
    return accuracy, model

if __name__ == "__main__":
    train() 