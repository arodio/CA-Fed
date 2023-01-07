import torch

from copy import deepcopy


def average_models(
        trainers,
        target_trainer,
        weights=None,
        average_params=True,
        average_gradients=False
):
    r"""computes the average of trainers and store it into target_trainer

    Parameters
    ----------
    trainers: List[Trainer]

    target_trainer: Trainer

    weights: 1-D torch.tensor
        tensor of the same size as trainers, having values between 0 and 1, and summing to 1,
        if not provided, uniform weights are used

    average_params: bool
        if set to true the parameters are averaged; default is True

    average_gradients: bool
        if set to true the gradient are averaged; default is False

    Returns
    -------
        None
    """
    if not average_params and not average_gradients:
        return

    if weights is None:
        n_trainers = len(trainers)
        weights = (1 / n_trainers) * torch.ones(n_trainers, device=trainers[0].device)

    else:
        weights = weights.to(trainers[0].device)

    param_tensors = []
    grad_tensors = []

    for trainer in trainers:
        if average_params:
            param_tensors.append(deepcopy(trainer.get_param_tensor()))

        if average_gradients:
            grad_tensors.append(deepcopy(trainer.get_grad_tensor()))

    if average_params:
        param_tensors = torch.stack(param_tensors)
        average_params_tensor = weights @ param_tensors
        target_trainer.set_param_tensor(average_params_tensor)

    if average_gradients:
        grad_tensors = torch.stack(grad_tensors)
        average_grads_tensor = weights @ grad_tensors
        target_trainer.set_grad_tensor(average_grads_tensor)


def copy_model(target, source):
    """copy learners_weights from target to source


    Parameters
    ----------
    target: torch.nn.Module

    source: torch.nn.Module


    Returns
    -------
        None

    """
    target.load_state_dict(source.state_dict())
