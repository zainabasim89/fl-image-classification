from .fedavg import FedAvgTrainer
from .fedprox import FedProxTrainer
from .fedavgm import FedAvgMTrainer
from .fedadam import FedAdamTrainer
from .fedadagrad import FedAdagradTrainer
from .fedyogi import FedYogiTrainer
from .fedmedian import FedMedianTrainer
from .fednova import FedNovaTrainer
from .fedbn import FedBNTrainer
from .fedpara import FedParaTrainer

TRAINER_REGISTRY = {
    "fedavg": FedAvgTrainer,
    "fedprox": FedProxTrainer,
    "fedavgm": FedAvgMTrainer,
    "fedadam": FedAdamTrainer,
    "fedadagrad": FedAdagradTrainer,
    "fedyogi": FedYogiTrainer,
    "fedmedian": FedMedianTrainer,
    "fednova": FedNovaTrainer,
    "fedbn": FedBNTrainer,
    "fedpara": FedParaTrainer,
}


def get_trainer(algorithm: str, **kwargs):
    try:
        trainer_cls = TRAINER_REGISTRY[algorithm]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Available: {list(TRAINER_REGISTRY)}"
        )
    return trainer_cls(**kwargs)