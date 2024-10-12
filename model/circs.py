import os
import pdb
import pickle
import random
import sys
sys.path.append(os.getcwd() + '/data_preparation')
from helper import get_path_training_data
from typing import Callable, List, Optional, Tuple, Any, TYPE_CHECKING

import logging
logger = logging.getLogger("qet-predictor")

if sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources  # type: ignore[no-redef]

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
import scipy.signal
import torch
from torchpack.datasets.dataset import Dataset

__all__ = ["CircDataset", "Circ"]

if sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources  # type: ignore[no-redef]

if TYPE_CHECKING:
    from numpy._typing import NDArray


random.seed(1234)


def load_training_data():
    """Loads and returns the training data from the training data folder.
    """
    file = open(get_path_training_data() / "training_data_standardization.npy", "rb")
    # file = open(get_path_training_data() / "washington_standardization.npy", "rb")
    # file = open(get_path_training_data() / "sherbrooke_standardization.npy", "rb")
    training_data = pickle.load(file)
    file.close()
    return training_data


class CircDataset:
    def __init__(self, split_ratio: List[float], shuffle=True):
        super().__init__()
        self.split_ratio = split_ratio
        self.raw = {}
        self.mean = {}
        self.std = {}

        self.shuffle = shuffle

        self._load()
        self._preprocess()
        self._split()

        self.instance_num = len(self.raw["dataset"])
        

    def _load(self):
        self.raw["dataset"] = load_training_data()
        if self.shuffle:
            random.shuffle(self.raw["dataset"])
        for data in self.raw["dataset"]:
            data.global_features = data.global_features.unsqueeze(0)


    def _preprocess(self):
        pass
        

    def _split(self):
        instance_num = len(self.raw["dataset"])
        split_train = self.split_ratio[0]
        split_valid = self.split_ratio[0] + self.split_ratio[1]

        self.raw["train"] = self.raw["dataset"][: int(split_train * instance_num)]
        self.raw["valid"] = self.raw["dataset"][
            int(split_train * instance_num) : int(split_valid * instance_num)
        ]
        self.raw["test"] = self.raw["dataset"][int(split_valid * instance_num) :]
        # self.raw["test"] = self.raw["dataset"][int(split_train * instance_num) :]

    def get_data(self, device, split):
        return [data.to(device) for data in self.raw[split]]

    def __getitem__(self, index: int):
        data_this = {"dag": self.raw["dataset"][index]}
        return data_this

    def __len__(self) -> int:
        return self.instance_num


class Circ(Dataset):
    def __init__(
        self,
        root: str,
        split_ratio: List[float]
    ):
        self.root = root

        super().__init__(
            {
                split: CircDataset(
                    root=root,
                    split=split,
                    split_ratio=split_ratio,
                )
                for split in ["train", "valid", "test"]
                # for split in ["train", "test"]
            }
        )

