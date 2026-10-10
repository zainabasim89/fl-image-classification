from abc import ABC, abstractmethod
import torch

class BaseTrainer(ABC):
    """Common interface every algorithm trainer must implement."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def compute_loss(self, model, outputs, labels, criterion) -> torch.Tensor:
        """Return the (possibly augmented) loss for a single batch."""
        raise NotImplementedError

    def on_train_start(self, model):
        """Hook called once before local training begins.
        Override if the algorithm needs to snapshot state (e.g. global params)."""
        pass

    def train(self, model, trainloader, epochs, lr, device):
        """Shared training loop — same for every algorithm.
        Only compute_loss / on_train_start differ per strategy."""
        self.on_train_start(model)

        model.to(device)
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        total_loss = 0.0
        num_batches = 0
        for _ in range(epochs):
            for batch in trainloader:
                images, labels = batch["img"].to(device), batch["label"].to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = self.compute_loss(model, outputs, labels, criterion)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches