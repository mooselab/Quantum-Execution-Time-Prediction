from __future__ import annotations

import os
import sys
sys.path.append(os.getcwd() + '/data_preparation')
from typing import Any
import logging
logger = logging.getLogger("qet-predictor")

if sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources  # type: ignore[no-redef]

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from joblib import dump, Parallel, delayed, load
from utils import calc_supermarq_features
from qiskit import QuantumCircuit, Aer, execute, transpile


if TYPE_CHECKING:
    from numpy._typing import NDArray

import string

import networkx as nx
import rustworkx as rx
import torch
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGInNode, DAGOpNode, DAGOutNode
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeToronto, FakeWashington, FakeMontreal, FakeTokyo, FakeSherbrooke 
from qiskit.transpiler.passes import RemoveBarriers, RemoveFinalMeasurements 
from torch_geometric.utils.convert import from_networkx
from qiskit_ibm_provider import IBMProvider
import json
import pickle
import pandas as pd
from pathlib import Path
import os
import shutil
import queue
import copy
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score


QISKIT_TOKEN = 'QISKIT_TOKEN' # input your own token: https://quantum.ibm.com/account
IBM_PROVIDER = IBMProvider(token=QISKIT_TOKEN)



provider_dict = {
    'TORONTO': {'provider': FakeToronto(), 'max_qubits': 27},
    'WASHINGTON': {'provider': FakeWashington(), 'max_qubits': 127},
    'MONTREAL': {'provider': FakeMontreal(), 'max_qubits': 27},
    'TOKYO': {'provider': FakeTokyo(), 'max_qubits': 20},
    'SHERBROOKE': {'provider': FakeSherbrooke(), 'max_qubits': 127},
    # 'OSAKA': {'provider': IBM_PROVIDER.get_backend('ibm_osaka'), 'max_qubits': 127},
    # 'KYOTO': {'provider': IBM_PROVIDER.get_backend('ibm_kyoto'), 'max_qubits': 127},
    # 'BRISBANE': {'provider': IBM_PROVIDER.get_backend('ibm_brisbane'), 'max_qubits': 127}
}

def get_path_training_data():
    """Returns the path to the training data folder."""
    return Path(os.getcwd()) / "data"


def get_openqasm_gates():
    """Returns a list of all quantum gates within the openQASM 2.0 standard header."""
    # according to https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/qasm/libs/qelib1.inc
    return [
        "u3",
        "u2",
        "u1",
        "cx",
        "id",
        "u0",
        "u",
        "p",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "rx",
        "ry",
        "rz",
        "sx",
        "sxdg",
        "cz",
        "cy",
        "swap",
        "ch",
        "ccx",
        "cswap",
        "crx",
        "cry",
        "crz",
        "cu1",
        "cp",
        "cu3",
        "csx",
        "cu",
        "rxx",
        "rzz",
        "rccx",
        "rc3x",
        "c3x",
        "c3sqrtx",
        "c4x",
        "xx_plus_yy",
        "ecr",
    ]


def dict_to_featurevector(gate_dict: dict[str, int]):
    """Calculates and returns the feature vector of a given quantum circuit gate dictionary."""
    res_dct = dict.fromkeys(get_openqasm_gates(), 0)
    for key, val in dict(gate_dict).items():
        if key in res_dct:
            res_dct[key] = val

    return res_dct


PATH_LENGTH = 260


def create_feature_dict(qc: str | QuantumCircuit):
    """Creates and returns a feature dictionary for a given quantum circuit.

    Args:
        qc (str | QuantumCircuit): The quantum circuit to be compiled.

    Returns:
        dict[str, Any]: The feature dictionary of the given quantum circuit.
    """
    if not isinstance(qc, QuantumCircuit):
        if len(qc) < PATH_LENGTH and Path(qc).exists():
            qc = QuantumCircuit.from_qasm_file(qc)
        elif "OPENQASM" in qc:
            qc = QuantumCircuit.from_qasm_str(qc)
        else:
            error_msg = "Invalid input for 'qc' parameter."
            raise ValueError(error_msg) from None

    ops_list = qc.count_ops()
    ops_list_dict = dict_to_featurevector(ops_list)

    feature_dict = {}
    for key in ops_list_dict:
        feature_dict[key] = float(ops_list_dict[key])

    feature_dict["num_qubits"] = float(qc.num_qubits)
    feature_dict["depth"] = float(qc.depth())

    supermarq_features = calc_supermarq_features(qc)
    feature_dict["program_communication"] = supermarq_features.program_communication
    feature_dict["critical_depth"] = supermarq_features.critical_depth
    feature_dict["entanglement_ratio"] = supermarq_features.entanglement_ratio
    feature_dict["parallelism"] = supermarq_features.parallelism
    feature_dict["liveness"] = supermarq_features.liveness
    return feature_dict


def refine_training_data(
    sample
):
    training_data = copy.deepcopy(sample)
    global_features_list = []
    for td in training_data:
        # print(td.global_features)
        global_features_list.append(np.array(td.global_features))
    # print(global_features_list)
    global_features_list = np.array(global_features_list)
    non_zero_indices = []
    for i in range(len(global_features_list[0])):
        if sum(global_features_list[:, i]) > 0:
                non_zero_indices.append(i)
    for td in training_data:
        td.global_features = torch.tensor(np.array(td.global_features)[non_zero_indices])
    print(non_zero_indices)
    return training_data


def standardization_training_data(
    sample
):
    training_data = copy.deepcopy(sample)
    x, global_features = torch.tensor([]), torch.tensor([])
    for td in training_data:
        x = torch.cat([x, td.x])
        global_features = torch.cat([global_features, td.global_features])

    means_x = x.mean(0)
    stds_x = x.std(0)
    means_gf = global_features.mean(0)
    stds_gf = global_features.std(0)
    
    for td in training_data:
        td.x = (td.x - means_x) / (1e-8 + stds_x)
        td.global_features = (td.global_features - means_gf) / (
            1e-8 + stds_gf
        )
    return training_data


def normalize_training_data(
    sample
):
    training_data = copy.deepcopy(sample)
    x, global_features = torch.tensor([]), torch.tensor([])
    for td in training_data:
        x = torch.cat([x, td.x])
        global_features = torch.cat([global_features, td.global_features])

    maxs_x = x.max(0)[0]
    mins_x = x.min(0)[0]
    maxs_gf = global_features.max(0)[0]
    mins_gf = global_features.min(0)[0]

    for td in training_data:
        td.x = (td.x - mins_x) / (1e-8 + maxs_x - mins_x)
        td.global_features = (td.global_features - mins_gf) / (
            1e-8 + maxs_gf - mins_gf
        )
    return training_data


def save_training_data(
    training_data: list[NDArray[np.float_]]
):
    """Saves the given training data to the training data folder.
    """
    file = open(get_path_training_data() / "training_data.npy", "wb")
    pickle.dump(training_data, file)
    file.close()


def load_training_data():
    """Loads and returns the training data from the training data folder.
    """
    file = open(get_path_training_data() / "training_data.npy", "rb")
    training_data = pickle.load(file)
    file.close()
    return training_data


def calc(y_true, y_pred):
    # calculate R-squared
    r_squared = r2_score(y_true, y_pred)
    
    # calculate MSE
    mse = mean_squared_error(y_true, y_pred)
    
    # calculate NMSE
    nmse = mse / np.var(y_true)
    
    print("MSE:", mse)
    print("R-squared:", r_squared)
    print("NMSE:", nmse)