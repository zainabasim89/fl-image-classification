from .base import BaseTrainer


class FedAdagradTrainer(BaseTrainer):
    """Client-side trainer for FedAdagrad."""

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)