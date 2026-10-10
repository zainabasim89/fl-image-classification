from .base_trainer import BaseTrainer


class FedAvgTrainer(BaseTrainer):
    """Plain local training — no extra regularization."""

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)