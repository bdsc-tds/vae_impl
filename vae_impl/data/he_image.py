import os
import numpy as np
from PIL import Image
from typing import Optional

import torch
import lightning.pytorch as pl
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class HEImageDataset(Dataset):
    def __init__(self, dir: str, transform: Optional[callable] = None) -> None:
        """Constructor for HEImageDataset.

        Args:
            dir (str): Path to the folder containing the slices of an H&E image.
            transform (Optional[callable], optional): Transform functions for the images. Defaults to None, where images will be automatically normalized in [0, 1].
        """

        self.dir = dir
        self.filenames = os.listdir(dir)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = lambda x: transforms.ToTensor()(np.array(x))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[Tensor, str]:
        image = self.transform(
            Image.open(os.path.join(self.dir, self.filenames[idx])).convert("RGB")
        )

        label = (
            self.filenames[idx]
            if "." not in self.filenames[idx]
            else self.filenames[idx].split(".")[0]
        )
        return image, label


class HEImageDataModule(pl.LightningDataModule):
    def __init__(
            self,
            dir: dir,
            num_workers: int = 0,
            train_val_test: tuple[int | float, int | float, int | float] = (0.6, 0.2, 0.2),
            batch_size: int = 32,
            shuffle: bool = True,
            auto_normalize: bool = True,
            mean: tuple[int | float] = (128, 128, 128),
            std: tuple[int | float] = (128, 128, 128),
            pin_memory: bool = True,
            drop_last: bool = False,
    ) -> None:
        """Constructor for HEImageDataModule.

        Args:
            dir (dir): Path to the folder containing the slices of an H&E image.
            num_workers (int, optional): Number of workers (for DataLoader). Defaults to 0.
            train_val_test (tuple[int  |  float, int  |  float, int  |  float], optional): Proportions or numbers of images for training, validation and test set. Defaults to (0.6, 0.2, 0.2).
            batch_size (int, optional): Batch size for the training set (for DataLoader). Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the training set for each epoch (for DataLoader). Defaults to True.
            auto_normalize (bool, optional): Whether to normalize the images automatically in training, validation and test set. Defaults to True.
            mean (tuple[int  |  float], optional): The mean used to normalize the images. Only valid when `auto_normalize` is False. Defaults to (128, 128, 128).
            std (tuple[int  |  float], optional): The standard deviation used to normalize the images. Only valid when `auto_normalize` is False. Defaults to (128, 128, 128).
            pin_memory (bool, optional): Whether to pin memory (for DataLoader). Defaults to True.
            drop_last (bool, optional): Whether to drop the last batch in the training set (for DataLoader). Defaults to False.
        """

        super().__init__()

        self.auto_normalize = auto_normalize
        self.normalize_mean = mean
        self.normalize_std = std

        self.dataset = HEImageDataset(dir, self.default_transforms())

        self.train_val_test = tuple([i / sum(train_val_test) for i in train_val_test])
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def default_transforms(self) -> Optional[callable]:
        if self.auto_normalize:
            return None

        return lambda x: transforms.Normalize(self.normalize_mean, self.normalize_std)(
            torch.tensor(np.transpose(np.asarray(x, dtype=np.float32), (2, 0, 1)))
        )

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset_train, self.dataset_val, self.dataset_test = random_split(
            self.dataset, self.train_val_test, generator=torch.Generator()
        )

    def train_dataloader(self, *args: any, **kwargs: any) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self, *args: any, **kwargs: any) -> DataLoader:
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self, *args: any, **kwargs: any) -> DataLoader:
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def get_normalization(self):
        if self.auto_normalize:
            return None

        return self.normalize_mean, self.normalize_std
