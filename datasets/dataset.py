import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import transforms as T

from PIL import Image
from functools import partial
from pathlib import Path

from utils.utils import function_convert_image_to, identity

class Dataset(Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png', 'tiff'], augment_horizontal_flip = False, convert_image_to = None):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(function_convert_image_to, convert_image_to) if convert_image_to is not None else identity

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)