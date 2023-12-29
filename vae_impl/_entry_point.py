from typing import Any, Optional

import lightning.pytorch as pl
import torch
import numpy as np
import pandas as pd

import vae_impl._constants as consts
from vae_impl.utils import get_latent_feats
from vae_impl.model.vae.base_vae import BaseVAE
from vae_impl.model.vae.cnn_vae import CNNVAE
from vae_impl.data.he_image import HEImageDataModule
from vae_impl.data.mnist import MNISTDataModule

# from vae_impl.data.cifar10 import CIFAR10DataModule
# from vae_impl.callbacks.fine_tune_learning_rate_finder import FineTuneLearningRateFinder


def entry_point(
    seed: int,
    data: str | type[pl.LightningDataModule],
    data_args: dict[str, Any],
    model: str | type[BaseVAE],
    model_args: dict[str, Any],
    use_early_stop: bool = True,
    early_stop_args: dict[str, Any] = {
        "monitor": "val_loss",
        "mode": "min",
        "min_delta": 0.0,
        "patience": 5,
        "verbose": True,
    },
    trainer_args: dict[str, Any] = {
        "accelerator": "auto",
        "devices": 1,
        "min_epochs": 1,
        "max_epochs": 100,
    },
    default_root_dir: Optional[str] = None,
    to_dataframe: bool = True,
) -> tuple[type[BaseVAE], pd.DataFrame | tuple[np.ndarray, list[str]]]:
    # seed
    pl.seed_everything(seed)

    # prepare data
    if isinstance(data, str):
        if data in ["h", "he", "he_image"]:
            data = HEImageDataModule(**data_args)
        elif data in ["m", "mnist"]:
            data = MNISTDataModule(**data_args)
        # elif data in ["c10", "cifar10"]:
        #     data = CIFAR10DataModule(**data_args)
        else:
            raise RuntimeError(f"Unsupported data type: {data}")

    # prepare model
    if isinstance(model, str):
        if model in ["cnn"]:
            model = CNNVAE(**model_args).to(consts.DEVICE)
        else:
            raise RuntimeError(f"Unsupported model type: {model}")

    # use early stop
    if use_early_stop:
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            **early_stop_args
        )

        if "callbacks" in trainer_args.keys():
            trainer_args.pop("trainer_args")

    trainer_args["default_root_dir"] = default_root_dir

    # prepare trainer
    trainer = pl.Trainer(
        **trainer_args,
        callbacks=[early_stop_callback] if use_early_stop else None,
    )

    # train the model
    trainer.fit(model, datamodule=data)

    # test the model
    trainer.test(model, datamodule=data)

    return model, get_latent_feats(model, data, to_dataframe)
