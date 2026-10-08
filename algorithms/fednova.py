"""FedNova: Normalized Averaging for Heterogeneous Clients (Wang et al., NeurIPS 2020)."""

import numpy as np
from flwr.app import Array, ArrayRecord
from flwr.serverapp.strategy import FedAvg

from .base import BaseTrainer


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

class FedNova(FedAvg):
    """Server strategy: normalizes client updates by local steps (a_i) with optional server momentum."""

    def __init__(
        self,
        server_learning_rate: float = 1.0,
        server_momentum: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.server_learning_rate = 1.0 if server_learning_rate is None else float(server_learning_rate)
        self.server_momentum = 0.0 if server_momentum is None else float(server_momentum)
        self.momentum_buffer = None

    def configure_train(self, server_round, arrays, config, grid):
        self.current_arrays = arrays
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True, validate=False)
        if not valid_replies:
            return None, None

        global_ndarrays = self.current_arrays.to_numpy_ndarrays()
        array_keys = list(self.current_arrays.keys())

        # Collect sample counts n_i, local steps a_i, and client weights x_i
        weights = [float(msg.content["metrics"].get(self.weighted_by_key, 1.0)) for msg in valid_replies]
        local_norms = [float(msg.content["metrics"].get("local_norm", 1.0)) for msg in valid_replies]
        client_ndarrays = [msg.content["arrays"].to_numpy_ndarrays() for msg in valid_replies]

        # Dataset proportions p_i = n_i / N
        total_w = sum(weights) or 1.0
        proportions = [w / total_w for w in weights]

        # Effective steps tau_eff = sum(p_i * a_i) (Wang et al., Eq. 6)
        tau_eff = sum(p * a for p, a in zip(proportions, local_norms))
        scales = [p * (tau_eff / a) for p, a in zip(proportions, local_norms)]

        # Normalized updates: sum(scale_i * (old - client[i]))
        updates = [
            sum(s * (old - client[i]) for s, client in zip(scales, client_ndarrays))
            for i, old in enumerate(global_ndarrays)
        ]

        # Apply server momentum if configured (matching FedNova paper / baseline)
        if self.server_momentum > 0.0:
            if self.momentum_buffer is None:
                self.momentum_buffer = [np.zeros_like(u) for u in updates]
            self.momentum_buffer = [
                self.server_momentum * m + u for m, u in zip(self.momentum_buffer, updates)
            ]
            applied_updates = self.momentum_buffer
        else:
            applied_updates = updates

        # Apply server learning rate: x_{t+1} = x_t - server_learning_rate * applied_updates
        new_ndarrays = [
            old - self.server_learning_rate * u
            for old, u in zip(global_ndarrays, applied_updates)
        ]

        self.current_arrays = ArrayRecord({k: Array(np.asarray(v)) for k, v in zip(array_keys, new_ndarrays)})
        metrics = self.train_metrics_aggr_fn([msg.content for msg in valid_replies], self.weighted_by_key)
        return self.current_arrays, metrics