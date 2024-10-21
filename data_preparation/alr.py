import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import random
import argparse
import pandas as pd
import numpy as np
import pickle
import shutil
import copy

import torch
import torch.nn.functional as F
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from qiskit import QuantumCircuit

import os
from pathlib import Path

import sys
sys.path.append(os.getcwd() + '/data_preparation')
import helper
import circ_dag_converter

def generate_active_learning_training_sample():
    source_path = str(helper.get_path_training_data() / 'quantum_circuits')
    folder = Path(source_path)
    final_result = []
    for filename in folder.iterdir():
        if filename.suffix != ".qasm":
            continue
        filename = str(filename).split("/")[-1]
        for device in ['osaka', 'kyoto']:
            qc = QuantumCircuit.from_qasm_file(source_path + '/' + filename)
            global_features = helper.create_feature_dict(qc)
            circ_graph_feature = circ_dag_converter.circ_to_dag_with_data(qc, device, list(global_features.values()), n_qubit=127)
            final_result.append([filename, device, circ_graph_feature])
    result_path = str(helper.get_path_training_data() / 'osaka_kyoto.npy')
    with open(result_path, "wb") as file:
        pickle.dump(final_result, file)


def load_active_learning_training_sample():
    with open(str(helper.get_path_training_data() / 'osaka_kyoto.npy'), "rb") as file:
        data = pickle.load(file)
    return data


def refine_training_data_active_learning(
    sample
):
    training_data = copy.deepcopy(sample)
    global_features_list = []
    for td in training_data:
        global_features_list.append(np.array(td[2].global_features))
    # print(global_features_list)
    global_features_list = np.array(global_features_list)
    non_zero_indices = []
    for i in range(len(global_features_list[0])):
        if sum(global_features_list[:, i]) > 0:
                non_zero_indices.append(i)
    for td in training_data:
        td[2].global_features = torch.tensor(np.array(td[2].global_features)[non_zero_indices])
    return training_data


def standardization_training_data_active_learning(
    sample
):
    training_data = copy.deepcopy(sample)
    x, global_features = torch.tensor([]), torch.tensor([])
    for td in training_data:
        x = torch.cat([x, td[2].x])
        global_features = torch.cat([global_features, td[2].global_features])
    
    means_x = x.mean(0)
    stds_x = x.std(0)
    means_gf = global_features.mean(0)
    stds_gf = global_features.std(0)
    
    for td in training_data:
        td[2].x = (td[2].x - means_x) / (1e-8 + stds_x)
        td[2].global_features = (td[2].global_features - means_gf) / (
            1e-8 + stds_gf
        )
    return training_data


def normalize_training_data_active_learning(
    sample
):
    training_data = copy.deepcopy(sample)
    x, global_features = torch.tensor([]), torch.tensor([])
    for td in training_data:
        x = torch.cat([x, td[2].x])
        global_features = torch.cat([global_features, td[2].global_features])

    maxs_x = x.max(0)[0]
    mins_x = x.min(0)[0]
    maxs_gf = global_features.max(0)[0]
    mins_gf = global_features.min(0)[0]

    for td in training_data:
        td[2].x = (td[2].x - mins_x) / (1e-8 + maxs_x - mins_x)
        td[2].global_features = (td[2].global_features - mins_gf) / (
            1e-8 + maxs_gf - mins_gf
        )
    return training_data


def padding_training_data_active_learning(
    sample
):
    training_data = copy.deepcopy(sample)
    max_dimension = 0
    for td in training_data:
        max_dimension = max(max_dimension, td[2].x.shape[0])
    print(max_dimension)
    for td in training_data:
        td[2].x = F.pad(td[2].x, (0, 0, 0, max_dimension - td[2].x.shape[0]), "constant", 0)

    return training_data


def GX(sample, num):
    x = torch.zeros_like(sample[0][2].x)
    gf = torch.zeros_like(sample[0][2].global_features)
    result = {}
    index_list = []
    for data in sample:
        x += data[2].x
        gf += data[2].global_features
    centroid = [x/len(sample), gf/len(sample)]
    dist = np.inf
    index = 0
    for j, data in enumerate(sample):
        d = 0.5 * torch.linalg.norm(data[2].x - centroid[0]) \
            + 0.5 * torch.linalg.norm(data[2].global_features - centroid[1])
        if d < dist:
            index, dist = j, d
    index_list.append(index)

    while len(index_list) < num:
        dist = [np.inf] * len(sample)
        for j, data in enumerate(sample):
            if j in index_list:
                dist[j] = 0
                continue
            for k in index_list:
                d = 0.5 * torch.linalg.norm(data[2].x - sample[k][2].x) \
                    + 0.5 * torch.linalg.norm(data[2].global_features - sample[k][2].global_features)
                if d < dist[j]:
                    dist[j] = d
        index_list.append(np.argmax(dist))
    
    sample_list = [sample[i] for i in index_list]
    for sample in sample_list:
        if sample[1] not in result:
            result[sample[1]] = []
        result[sample[1]].append(sample[0])
    
    return result


def generate_active_learning_samples(circuit_dict):
    source_dir = str(helper.get_path_training_data() / 'quantum_circuits')

    for computer, circuit_list in circuit_dict.items():
        destination_dir = source_dir + '_' + computer
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        for circuit in circuit_list:
            destination_file = os.path.join(destination_dir, circuit)
            source_file = os.path.join(source_dir, circuit)
            shutil.copy(source_file, destination_file)
