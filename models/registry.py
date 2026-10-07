from .base import BaseModel
from .mobilenet import MobileNetModel
from .resnet import ResNetModel
from .cnn import CNNModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "mobilenet": MobileNetModel,
    "resnet": ResNetModel,
    "cnn": CNNModel,
}