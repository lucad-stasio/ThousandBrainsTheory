class CNN_Monolith(nn.Module):
    def __init__(self):
        super(CNN_Monolith, self).__init__()
        # Convolutional Layers looking for 2D patterns
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # The massive Dense layer processing the flattened grid
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 10 possible digits
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7) # Flattening the 2D grid
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn_model = CNN_Monolith().to(device)
print(f"CNN Parameters: {sum(p.numel() for p in cnn_model.parameters())}")
