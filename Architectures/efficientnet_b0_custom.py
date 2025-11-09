import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class EfficientNetB0Custom(nn.Module):
    """
    EfficientNet-B0 adapted for GTSRB.
    Compact yet powerful model for efficiency–accuracy tradeoff analysis.
    """
    def __init__(self, num_classes=43, pretrained=False):
        super().__init__()
        # Load EfficientNet-B0 backbone
        self.model = efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        # Replace classifier head
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
