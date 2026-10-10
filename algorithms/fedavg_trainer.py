from .base import BaseTrainer


class FedAvgTrainer(BaseTrainer):
    """Plain local training — no extra regularization."""

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)