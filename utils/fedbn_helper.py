from collections import OrderedDict

import torch.nn as nn


def get_bn_keys(model):
    """Return state_dict keys belonging to BatchNorm layers."""
    bn_keys = set()

    for module_name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            prefix = f"{module_name}."

            for key in model.state_dict().keys():
                if key.startswith(prefix):
                    bn_keys.add(key)

    return bn_keys


def get_fedbn_state_dict(model):
    """Return model state excluding BatchNorm state."""
    bn_keys = get_bn_keys(model)

    return OrderedDict(
        (key, value)
        for key, value in model.state_dict().items()
        if key not in bn_keys
    )


def get_bn_state_dict(model):
    """Return only BatchNorm state."""
    bn_keys = get_bn_keys(model)

    return OrderedDict(
        (key, value)
        for key, value in model.state_dict().items()
        if key in bn_keys
    )


def load_bn_state_dict(model, bn_state):
    """Load locally retained BatchNorm state."""
    if bn_state:
        model.load_state_dict(bn_state, strict=False)