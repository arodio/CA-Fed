import torch


def accuracy(y_pred, y):
    """computes classification accuracy

    Parameters
    ----------
    y_pred: torch.tensor

    y: 1-D torch.torch.tensor

    Returns
    -------
        * float

    """
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y).float()
    acc = correct.sum() / len(y)
    return acc


def binary_accuracy(y_pred, y):
    y_pred = torch.round(torch.sigmoid(y_pred))  # round predictions to the closest integer
    correct = (y_pred == y).float()
    acc = correct.sum() / len(y)
    return acc
