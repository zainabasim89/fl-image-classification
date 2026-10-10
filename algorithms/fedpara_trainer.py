from .base_trainer import BaseTrainer

class FedParaTrainer(BaseTrainer):
    """Client-side trainer for FedPara matching Flower's baseline."""

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)