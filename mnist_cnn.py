import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
from pathlib import Path
import random

# Create results directory if it doesn't exist
Path("results").mkdir(exist_ok=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and transform data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize model, loss, and optimizer
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 10
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                current_loss = epoch_loss / (batch_idx + 1)
                log_data = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': current_loss
                }
                with open('results/training_log.json', 'a') as f:
                    f.write(json.dumps(log_data) + '\n')
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    # Save model
    torch.save(model.state_dict(), 'results/mnist_cnn.pth')
    
    # Generate predictions for random samples
    model.eval()
    test_samples = []
    with torch.no_grad():
        for _ in range(10):
            idx = random.randint(0, len(test_dataset)-1)
            img, label = test_dataset[idx]
            img = img.unsqueeze(0).to(device)
            output = model(img)
            pred = output.argmax(dim=1).item()
            test_samples.append({
                'idx': idx,
                'true': label,
                'pred': pred
            })
    
    with open('results/test_samples.json', 'w') as f:
        json.dump(test_samples, f)

if __name__ == '__main__':
    train() 