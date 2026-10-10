from .base_trainer import BaseTrainer

class FedAvgMTrainer(BaseTrainer):
    """FedAvgM client-side trainer.

    FedAvgM uses ordinary local training on clients.
    Momentum is applied on the server during aggregation.
    """

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)