from .base_trainer import BaseTrainer

class FedMedianTrainer(BaseTrainer):
    """Client-side trainer for FedMedian."""

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)