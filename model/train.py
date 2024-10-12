import argparse
import pdb
import random

import sys
sys.path.append('../model')
import builder
from trainer import trainer

import os
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torch_geometric.loader import DataLoader
from torchpack.utils.config import configs
from torchpack.utils.logging import logger


def main() -> None:
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    configs.evalmode = False

    parser = argparse.ArgumentParser()
    parser.add_argument("parameter_name", metavar="FILE", help="config file")
    parser.add_argument("--load", action="store_true", help="config file")

    args, opts = parser.parse_known_args()
    parameter_path = Path(os.getcwd()) / f"model/parameter/{args.parameter_name}"

    configs.load(str(parameter_path / "config.yaml"), recursive=True)
    configs.update(opts)
    configs.parameter_name = args.parameter_name
    if configs.device == "gpu":
        device = torch.device("cuda")
    elif configs.device == "cpu":
        device = torch.device("cpu")

    if isinstance(configs.optimizer.lr, str):
        configs.optimizer.lr = eval(configs.optimizer.lr)

    logger.info(" ".join([sys.executable] + sys.argv))

    model = builder.make_model()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Size: {total_params}")
    dataflow = {}
    if not args.load:
        dataset = builder.make_dataset()
        for split in ["train", "valid", "test"]:
        # for split in ["train", "test"]:
            dataflow[split] = DataLoader(
                dataset.get_data(device, split), batch_size=configs.batch_size
            )

    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)

    my_trainer = trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=dataflow,
        configs=configs,
    )
    if not args.load:
        my_trainer.train()
        my_trainer.test()
        my_trainer.saveall()


if __name__ == "__main__":
    main()
