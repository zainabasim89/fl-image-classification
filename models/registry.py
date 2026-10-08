from .base import BaseModel
from .mobilenet import MobileNetModel
from .resnet import ResNetModel
from .cnn import CNNModel
from .vgg import VGGModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "mobilenet": MobileNetModel,
    "resnet": ResNetModel,
    "cnn": CNNModel,
    "vgg": VGGModel, 
}