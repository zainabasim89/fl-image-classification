import torch.nn as nn
from torchvision import models

from .base import BaseModel


class MobileNetModel(BaseModel):

    def build(self) -> nn.Module:
        model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )

        # Freeze backbone
        for param in model.features.parameters():
            param.requires_grad = False

        # Classification head
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 10),
        )

        return model