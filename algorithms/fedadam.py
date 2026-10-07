from .base import BaseTrainer


class FedAdamTrainer(BaseTrainer):
    """Client-side trainer for FedAdam."""

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)