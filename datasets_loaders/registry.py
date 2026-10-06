from .brain_tumor_dataset import BrainTumorLoader
from .cifar10_dataset import CIFAR10Loader


DATASET_REGISTRY = {
    "brain_tumor": BrainTumorLoader,
    "cifar10": CIFAR10Loader,
}