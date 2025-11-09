import torch
import torch.nn as nn
from torchvision.models import resnet50


class ResNet50Custom(nn.Module):
    """
    ResNet-50 modified for GTSRB (43 classes).
    Deep architecture suitable as a teacher model for knowledge distillation.
    """
    def __init__(self, num_classes=43, pretrained=False):
        super().__init__()
        # Load ResNet-50 backbone
        self.model = resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        # Replace the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
