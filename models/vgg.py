import torch.nn as nn
from torchvision import models
from .base import BaseModel


class VGGModel(BaseModel):

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes

    def build(self) -> nn.Module:
        model = models.vgg11(weights=None)

        # 1. Adaptive pooling to (1, 1):
        # Automatically handles ANY input image size (32x32, 64x64, 128x128, 224x224, etc.)
        # and always produces exactly 512 features for the classifier
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 2. Dynamic classification head:
        # Accepts any number of target classes (self.num_classes)
        model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, self.num_classes),
        )

        return model