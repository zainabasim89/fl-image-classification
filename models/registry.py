from .base import BaseModel
from .mobilenet import MobileNetModel
from .resnet import ResNetModel
from .resnet18 import ResNet18Model
from .resnet50 import ResNet50Model
from .cnn import CNNModel
from .vgg import VGGModel
from .resnet18_mlb import ResNet18MLBModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "mobilenet": MobileNetModel,
    "resnet": ResNetModel,
    "resnet18": ResNet18Model,
    "resnet50": ResNet50Model,
    "cnn": CNNModel,
    "vgg": VGGModel, 
    "resnet18_mlb": ResNet18MLBModel,
}
