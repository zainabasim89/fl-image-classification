from .fedavg_trainer import FedAvgTrainer
from .fedprox_trainer import FedProxTrainer
from .fedavgm_trainer import FedAvgMTrainer
from .fedadam_trainer import FedAdamTrainer
from .fedadagrad_trainer import FedAdagradTrainer
from .fedyogi_trainer import FedYogiTrainer
from .fedmedian_trainer import FedMedianTrainer
from .fednova import FedNovaTrainer
from .fedbn_trainer import FedBNTrainer
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