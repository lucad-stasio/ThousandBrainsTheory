import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

# Force PyTorch to use the GPU if Colab has one assigned
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Engine running on: {device}")

# Download and format the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoaders to feed the networks in batches of 64 images at a time
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
