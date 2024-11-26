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
        # First conv block
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(10)
        
        # Second conv block
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(10)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(10 * 5 * 5, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 10)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 10 * 5 * 5)
        x = torch.relu(self.bn3(self.fc1(x)))
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