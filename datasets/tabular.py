from torch.utils.data import Dataset


class Tabular(Dataset):
    """Tabular Dataset

     Attributes
    ----------
    data: torch.tensor

    targets: torch.tensor


    Methods
    -------
    __init__

    __len__

    __getitem__

    """
    def __init__(self, data, targets):

        self.data = data

        self.targets = targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (data, target) where target is index of the target class.
        """
        data, target = self.data[index], self.targets[index]

        return data, target
