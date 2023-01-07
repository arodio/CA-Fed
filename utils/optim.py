import torch.optim as optim

from .constants import *


def get_optimizer(optimizer_name, model, lr):
    """returns torch.optim.Optimizer given an optimizer name, a model and learning rate

    Parameters
    ----------
    optimizer_name: str
        possible are {"sgd"}

    model: torch.nn.Module

    lr: float


    Returns
    -------
        * torch.optim.Optimizer
    """

    if optimizer_name == "sgd":
        return optim.SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY
        )
    else:
        raise NotImplementedError(
            f"{optimizer_name} is not a possible optimizer name; possible are: 'sgd'"
        )
