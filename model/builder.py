import torch
import torch.nn
import torch.optim
from torch import nn
from torchpack.utils.config import configs
from torchpack.utils.typing import Optimizer, Scheduler

import os
import sys
sys.path.append(os.getcwd() + '/data_preparation')
from circs import CircDataset
from transformer_model import Simple_Model

__all__ = [
    "make_dataset",
    "make_model",
    "make_criterion",
    "make_optimizer",
    "make_scheduler",
]


def make_dataset_from(name):
    dataset = CircDataset(name, [0, 0, 1.0])
    return dataset


def make_dataset():
    dataset = CircDataset(configs.dataset.split_ratio)
    return dataset


def make_model() -> nn.Module:
    model = Simple_Model(configs.model)
    return model


def make_criterion() -> nn.Module:
    if configs.criterion.name == "mse":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError(configs.criterion.name)
    return criterion


def make_optimizer(model: nn.Module) -> Optimizer:
    if configs.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=configs.optimizer.lr,
            momentum=configs.optimizer.momentum,
            weight_decay=configs.optimizer.weight_decay,
        )
    elif configs.optimizer.name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay,
        )
    else:
        raise NotImplementedError(configs.optimizer.name)
    return optimizer


def make_scheduler(optimizer: Optimizer) -> Scheduler:
    if configs.scheduler.name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs.num_epochs, eta_min=0
        )
    elif configs.scheduler.name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1,
        )
    else:
        raise NotImplementedError(configs.scheduler.name)
    return scheduler
