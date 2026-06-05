import torch.nn as nn
import torch.nn.functional as F

class FashionModel(nn.Module):
    def __init__(self):
        super(FashionModel, self).__init__()
        # 2 convolutional layers
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels = 16, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        # 2 fully connected linear layers
        self.fc1 = nn.Linear(in_features = 32*7*7, out_features = 128)
        self.fc2 = nn.Linear(in_features = 128, out_features = 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) # convolution 1 + ReLU + max pool
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # convolution 2 + ReLU + max pool
        x = x.view(-1, 32*7*7) # flatten data into a vector
        # fully connected layers + ReLU activation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
