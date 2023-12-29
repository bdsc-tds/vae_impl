import torch
import lightning.pytorch as pl
from lightning.fabric.utilities.exceptions import MisconfigurationException

import numpy as np
import pandas as pd


def get_device(verbose: bool = True) -> str:
    info: dict[str, str | bool] = {}

    if torch.cuda.is_available():
        device = "cuda"

        info[f"support_{device}"] = torch.backends.cuda.is_built()
        info["avail_dev_num"] = str(torch.cuda.device_count())
        info["curr_dev_idx"] = str(torch.cuda.current_device())
        info["curr_dev_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    elif torch.backends.mps.is_available():
        device = "mps"
        info[f"support_{device}"] = torch.backends.mps.is_built()
    else:
        device = "cpu"

    if verbose:
        print(f"Using {device}")

        if len(info) > 0:
            for k, v in info.items():
                print(f"  {k}: {v}")

    return device


def get_latent_feats(
    model: type[pl.LightningModule],
    data: type[pl.LightningDataModule],
    to_dataframe: bool = True,
) -> pd.DataFrame | tuple[np.ndarray, list[str]]:
    dataloaders: list[torch.utils.data.DataLoader] = []
    _feats: list[torch.Tensor] = []
    labels: list[str] = []

    try:
        dataloaders.append(data.train_dataloader())
    except MisconfigurationException:
        print("Training set is not available for the input data.")

    try:
        dataloaders.append(data.val_dataloader())
    except MisconfigurationException:
        print("Validation set is not available for the input data.")

    try:
        dataloaders.append(data.test_dataloader())
    except MisconfigurationException:
        print("Test set is not available for the input data.")

    assert len(dataloaders) > 0

    for dl in dataloaders:
        for _, batch in enumerate(dl):
            __feats, __labels = model.get_latent(batch=batch)

            _feats.append(__feats)
            labels.extend(__labels)

    feats: np.ndarray = torch.cat(_feats).detach().numpy()

    if to_dataframe:
        return pd.DataFrame(
            data=feats,
            index=labels,
            columns=[f"feat_{i}" for i in range(feats.shape[1])],
        )

    return feats, labels
