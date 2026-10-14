from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )


class VisionModel(nn.Module):
    """Multi-label classifier for grayscale chest radiographs."""

    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 32), ConvBlock(32, 64), ConvBlock(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(1),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected [batch, 1, height, width], got {tuple(x.shape)}")
        
        return self.classifier(self.features(x))
