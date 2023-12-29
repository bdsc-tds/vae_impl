import os
import re
import numpy as np
import struct
from typing import Optional

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms


class MNISTDataset(Dataset):
    TRAIN = "train"
    TEST = "t10k"
    IMAGES = "images"
    LABELS = "labels"

    def __init__(self, dir: str, transform: Optional[callable] = None):
        filenames = os.listdir(dir)

        self.train_images = self.__decompress__(
            os.path.join(dir, MNISTDataset.file_search(filenames, True, True)), False
        )
        self.train_labels = self.__decompress__(
            os.path.join(dir, MNISTDataset.file_search(filenames, True, False)), True
        )
        self.test_images = self.__decompress__(
            os.path.join(dir, MNISTDataset.file_search(filenames, False, True)), False
        )
        self.test_labels = self.__decompress__(
            os.path.join(dir, MNISTDataset.file_search(filenames, False, False)), True
        )

        self.train_size = self.train_labels.shape[0]
        self.test_size = self.test_labels.shape[0]

        if transform is not None:
            self.transform = transform
        else:
            self.transform = lambda x: transforms.ToTensor()(np.array(x))

    @staticmethod
    def file_search(filenames: list[str], is_train: bool, is_images: bool):
        train_or_test = MNISTDataset.TRAIN if is_train else MNISTDataset.TEST
        images_or_labels = MNISTDataset.IMAGES if is_images else MNISTDataset.LABELS

        pat = re.compile(f"^(?=.*{train_or_test})(?=.*{images_or_labels}).+$")
        files = [i for i in filenames if re.search(pat, i)]

        assert len(files) == 1

        return files[0]

    def __decompress__(self, file: str, is_label: bool) -> np.ndarray:
        assert os.path.exists(file)

        with open(
            file,
            "rb",
        ) as f:
            _, size = struct.unpack(">II", f.read(8))

            if not is_label:
                nrows, ncols = struct.unpack(">II", f.read(8))

            # data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8)).newbyteorder(">")
            data = np.fromfile(f, dtype=np.dtype(np.uint8)).newbyteorder(">")

            if not is_label:
                # H * W * (C = 1)
                data = data.reshape((size, nrows, ncols, 1))

        return data

    def get_test_size(self) -> int:
        return self.test_size

    def __len__(self) -> int:
        """Get the number of images in the entire training set.
        This is mostly used for spliting into training and validation set.
        Images in the test set is not included in this process.

        Returns:
            int: the number of images in the entire training set
        """

        return self.train_size

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        if idx < self.train_size:
            image = self.transform(self.train_images[idx])
            label = self.train_labels[idx]
        else:
            image = self.transform(self.test_images[idx - self.train_size])
            label = self.test_labels[idx - self.train_size]

        return image, label


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dir: dir,
        num_workers: int = 0,
        train_val: tuple[int | float, int | float] = (0.8, 0.2),
        batch_size: int = 32,
        shuffle: bool = True,
        auto_normalize: bool = True,
        mean: int | float = 128,
        std: int | float = 128,
        pin_memory: bool = True,
        drop_last: bool = False,
    ) -> None:
        """Constructor for HEImageDataModule.

        Args:
            dir (dir): Path to the folder containing the downloaded MNIST dataset.
            num_workers (int, optional): Number of workers (for DataLoader). Defaults to 0.
            train_val (tuple[int  |  float, int  |  float], optional): Proportions or numbers of images for training and validation set. Defaults to (0.8, 0.2).
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

        self.dataset = MNISTDataset(dir, self.default_transforms())

        self.train_val = tuple([i / sum(train_val) for i in train_val])
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
        self.dataset_train, self.dataset_val = random_split(
            self.dataset, self.train_val, generator=torch.Generator()
        )

        self.dataset_test = Subset(
            self.dataset,
            range(len(self.dataset), len(self.dataset) + self.dataset.get_test_size()),
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
