import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
import os

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First conv: 28x28 -> 26x26 (no padding)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        # Pool: 26x26 -> 13x13
        
        # Second conv: 13x13 -> 11x11 (no padding)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3)
        # Pool: 11x11 -> 5x5
        self.pool = nn.MaxPool2d(2, 2)
        # self.dropout = nn.Dropout(0.2)
        
        # Fully connected layers
        # Input: 10 channels * 5 * 5 = 250 features
        self.fc1 = nn.Linear(10 * 5 * 5, 32)
        # Output: 10 classes (digits 0-9)
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 28x28 -> 13x13
        x = self.pool(torch.relu(self.conv2(x)))  # 13x13 -> 5x5
        x = x.view(-1, 10 * 5 * 5)  # Flatten: 10 channels * 5 * 5 = 250
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    # Set device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset with normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    
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