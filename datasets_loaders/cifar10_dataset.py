from datasets import load_dataset
from .base import BaseDatasetLoader


class CIFAR10Loader(BaseDatasetLoader):

    def load(self):
        dataset = load_dataset("uoft-cs/cifar10")

        return dataset