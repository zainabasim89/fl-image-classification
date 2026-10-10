from .base_trainer import BaseTrainer

class FedBNTrainer(BaseTrainer):
    """Client-side trainer for FedBN."""

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)