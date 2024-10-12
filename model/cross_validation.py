import argparse
import pdb
import random

import os
import sys
sys.path.append(os.getcwd() + '/data_preparation')
from helper import get_path_training_data

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

import pickle

def main() -> None:
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    configs.evalmode = False

    parser = argparse.ArgumentParser()
    parser.add_argument("parameter_name", metavar="FILE", help="config file")

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
    # with open(str(get_path_training_data() / "osaka_standardization.npy"), "rb") as file:
    # with open(str(get_path_training_data() / "kyoto_standardization.npy"), "rb") as file:
    with open(str(get_path_training_data() / "total_ibm_standardization.npy"), "rb") as file:
        dataset = pickle.load(file) 
    random.shuffle(dataset)

    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)

    start_index = 0
    instance_num = len(dataset)
    fold_size = instance_num // 10
    r_squared_list = []
    mse_list = []
    nmse_list = []
    for i in range(10):
        end_index = start_index + fold_size
        
        dataflow = {}
        dataflow['train'] = DataLoader(
                [data.to(device) for data in dataset[: start_index] + dataset[end_index: ]], batch_size=configs.batch_size
            )
        dataflow['valid'] = DataLoader(
                [data.to(device) for data in dataset[start_index: end_index]], batch_size=configs.batch_size
            )
        
        my_trainer = trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=dataflow,
        configs=configs,
        )
        my_trainer.model.load_state_dict(torch.load(str(parameter_path / "model.pth")))
        my_trainer.train()
        mse_list.append(my_trainer.best)
        nmse_list.append(my_trainer.nmse)
        r_squared_list.append(my_trainer.r_squared)
        start_index = end_index

    print("r_squared_list = ", r_squared_list)
    print("average_r_squared = ", np.mean(r_squared_list))
    print("mse_list = ", mse_list)
    print("average_mse = ", np.mean(mse_list))
    print("nmse_list = ", nmse_list)
    print("average_nmse = ", np.mean(nmse_list))


if __name__ == "__main__":
    main()
