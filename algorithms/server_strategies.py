from flwr.serverapp.strategy import FedAvg, FedProx, FedAvgM, FedAdam, FedAdagrad, FedYogi, FedMedian

STRATEGY_REGISTRY = {
    "fedavg": FedAvg,
    "fedprox": FedProx,
    "fedavgm": FedAvgM,
    "fedadam": FedAdam,
    "fedadagrad": FedAdagrad,
    "fedyogi": FedYogi,
    "fedmedian": FedMedian,
}


def get_strategy(algorithm: str, **kwargs):
    try:
        strategy_cls = STRATEGY_REGISTRY[algorithm]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Available: {list(STRATEGY_REGISTRY)}"
        )
    return strategy_cls(**kwargs)