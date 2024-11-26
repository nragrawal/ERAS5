import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from datetime import datetime
import os
from model.augmentation_utils import get_transforms, show_augmented_images, get_augmentation_examples

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
        self.fc1 = nn.Linear(10 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 10 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get transforms
    train_transform, basic_transform = get_transforms()
    
    # Load datasets
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Show augmentation examples
    test_dataset = datasets.MNIST('data', train=True, download=True, transform=basic_transform)
    orig_images, aug_images = get_augmentation_examples(test_dataset)
    show_augmented_images(orig_images, aug_images)
    
    # Initialize model and training components
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    # Training loop
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
        
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}')
    
    accuracy = 100. * correct / total
    print(f'Training Accuracy: {accuracy:.2f}%')
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), f'models/mnist_model_{timestamp}.pth')
    
    return accuracy, model

if __name__ == "__main__":
    train() 