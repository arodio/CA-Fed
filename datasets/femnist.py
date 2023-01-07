from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

DIM = 28 * 28


class SubFEMNIST(Dataset):
    """
    Constructs a subset of FEMNIST dataset corresponding to one client;
    Initialized with the path to a `.pt` file;
    `.pt` file is expected to hold a tuple of tensors (data, targets) storing the images and there corresponding labels.
    Attributes
    ----------
    transform
    data: iterable of integers
    targets
    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, data, targets):
        self.transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        self.data, self.targets = data, targets

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = np.uint8(img.numpy() * 255)
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img.view(DIM), target
