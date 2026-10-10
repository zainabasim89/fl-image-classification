"""FedNova: Normalized Averaging for Heterogeneous Clients (Wang et al., NeurIPS 2020)."""
from .base_trainer import BaseTrainer

class FedNovaTrainer(BaseTrainer):
    """Client trainer: runs local training and tracks step count (a_i)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.local_norm = 1.0

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)

    def train(self, model, trainloader, epochs, lr, device):
        # Reuses shared BaseTrainer.train()
        loss = super().train(model, trainloader, epochs, lr, device)
        # Total local steps a_i = epochs * batches
        self.local_norm = float(epochs * len(trainloader))
        return loss

    def get_metrics(self) -> dict:
        return {"local_norm": self.local_norm}