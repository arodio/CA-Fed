import os
import numpy as np

from PIL import Image

from torch.utils.data import Dataset


class MNIST(Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

     Attributes
    ----------
     root: string
        Root directory of dataset

     train: bool, optional
        If True, creates dataset from training set, otherwise
        creates from test set.

     transform: callable, optional
        A function/transform that takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``

    data: torch.tensor

    targets: torch.tensor

    Methods
    -------
    __init__

    __len__

    __getitem__

    """
    def __init__(self, root: str, train: bool, transform=None) -> None:

        self.root = root

        self.train = train  # training set or test set

        self.transform = transform

        if train:
            self.data = np.load(os.path.join(self.root, "train_data.npy"))
            self.targets = np.load(os.path.join(self.root, "train_targets.npy"))

        else:
            self.data = np.load(os.path.join(self.root, "test_data.npy"))
            self.targets = np.load(os.path.join(self.root, "test_targets.npy"))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)
            img = img.reshape(-1)  # flatten the image

        return img, target
