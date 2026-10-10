from .base import BaseTrainer

class FedYogiTrainer(BaseTrainer):
    """Client-side trainer for FedYogi."""

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)