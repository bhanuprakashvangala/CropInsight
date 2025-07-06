from torch.utils.data import Dataset
from sklearn.datasets import load_digits
from PIL import Image
import numpy as np


class DigitsDataset(Dataset):
    """Dataset wrapping sklearn digits images for PyTorch."""

    def __init__(self, transform=None):
        digits = load_digits()
        self.images = digits.images
        self.labels = digits.target
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = np.stack([img] * 3, axis=-1)  # convert to 3 channels
        img = Image.fromarray((img * 16).astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[idx])
        return img, label
