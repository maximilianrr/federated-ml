import torch.nn.functional as F
from torch import nn
import math


class VisionModel(nn.Module):
    """
    A simple 2D-CNN for depth-based hand position recognition.
    """

    def __init__(self, input_size=1296, num_classes=5):
        """
        Initialize the VisionModel.
        
        Args:
            input_size (int): The size of the input data.
            num_classes (int): The number of classes to predict.
        """
        super(VisionModel, self).__init__()

        self.image_dim = (int(math.sqrt(input_size)))

        # first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
         # add a dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5) 

        # fully connected layer
        # 36 x 36 image pooled twice (by 2) results in a 9 x 9 feature map, across 64 channels would that result in 64 * 9 * 9 = 5184 features.
        self.fc1 = nn.Linear(5184, 128)
        self.fc2 = nn.Linear(128, num_classes) 


    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        # Reshape to (batch_size, channels, height, width)
        x = x.view(-1, 1, self.image_dim, self.image_dim)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten the tensor
        x = x.reshape(x.shape[0], -1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.fc2(x)
        return x