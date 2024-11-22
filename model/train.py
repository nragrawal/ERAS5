import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
import os

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First conv: 28x28 -> 26x26
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        # Pool: 26x26 -> 13x13
        # Second conv: 13x13 -> 11x11
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3)
        # Pool: 11x11 -> 5x5 (rounded down from 5.5)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        # Therefore final feature map is 8 channels of 5x5
        self.fc1 = nn.Linear(10 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Conv -> ReLU -> Pool
        x = self.pool(torch.relu(self.conv2(x)))  # Conv -> ReLU -> Pool
        x = x.view(-1, 10 * 5 * 5)  # Flatten: 8 channels * 5 * 5
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
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
    
    # Save model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), f'models/mnist_model_{timestamp}.pth')
    
    return accuracy, model

if __name__ == "__main__":
    train() 