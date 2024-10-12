from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import os
import sys
sys.path.append(os.getcwd() + '/data_preparation')
import helper
import circ_dag_converter

import matplotlib.pyplot as plt
import numpy as np
import pandas as  pd
import pickle
import shutil
from joblib import Parallel, delayed, load
import utils
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.compiler import transpile
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeToronto, FakeWashington, FakeMontreal, FakeTokyo, FakeSherbrooke
from qiskit_ibm_provider import IBMProvider


if TYPE_CHECKING:
    from numpy._typing import NDArray

plt.rcParams["font.family"] = "Times New Roman"

logger = logging.getLogger("qet-predictor")


class Execution:
    def __init__(self, logger_level: int = logging.INFO) -> None:
        logger.setLevel(logger_level)

    def calculate_execution_time(self, backend, timeout):
        filefolder = helper.get_path_training_data() / "quantum_circuits"
        provider = helper.provider_dict[backend.upper()]['provider']
        max_qubits = helper.provider_dict[backend.upper()]['max_qubits']
        circ = QuantumCircuit()
        result_dir = str(helper.get_path_training_data() / backend) + "/"
        for file in filefolder.iterdir():
            if file.suffix != ".qasm":
                continue
            print(file)
            data = pd.DataFrame(columns=["quantum_circuit", "time_taken"])
            execute_result = utils.timeout_watcher(self.execute_circuit, [circ, file, provider, max_qubits], timeout)
            if execute_result is not None and execute_result is not False:
                result = execute_result[0]
                qc = execute_result[1]
                new_row = {'quantum_circuit': qc.qasm(), 'time_taken': result.time_taken}
                data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
                print('result.time_taken = ', result.time_taken)
                data.to_csv(result_dir + file.name.split("/")[-1].split(".qasm")[0] + ".csv", index=False)
                new_file = str(helper.get_path_training_data() / "quantum_circuits_new") + "/" + file.name.split("/")[-1]
                shutil.move(file, new_file)


    def calculate_execution_time_real_device(self, backend):
        filefolder = helper.get_path_training_data() / ("quantum_circuits_" + backend)
        provider = helper.provider_dict[backend.upper()]['provider']
        max_qubits = helper.provider_dict[backend.upper()]['max_qubits']
        circ = QuantumCircuit()
        result_dir = str(helper.get_path_training_data() / backend) + "/"
        for file in filefolder.iterdir():
            if file.suffix != ".qasm":
                continue
            print(file)
            data = pd.DataFrame(columns=["quantum_circuit", "time_taken"])
            execute_result = self.execute_circuit(circ, file, provider, max_qubits)
            if execute_result is not None and execute_result is not False:
                result = execute_result[0]
                qc = execute_result[1]
                new_row = {'quantum_circuit': qc.qasm(), 'time_taken': result.time_taken}
                data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
                print('result.time_taken = ', result.time_taken)
                data.to_csv(result_dir + file.name.split("/")[-1].split(".qasm")[0] + ".csv", index=False)
                new_file = str(helper.get_path_training_data() / ("quantum_circuits_" + backend + "_new")) + "/" + file.name.split("/")[-1]
                shutil.move(file, new_file)


    def execute_circuit(self, circ, file, provider, max_qubits):
        try:
            qc = circ.from_qasm_file(file)
        except BaseException as e:
            logger.warning(e)
            return False
        if qc.num_qubits > max_qubits:
            return False
        try:
            job = execute(qc, provider, shots=1024)
            print('finish execution')
        except BaseException as e:
            logger.warning(e)
            return False
        try:
            result = job.result()
            print('get result')
        except RuntimeError as e:
            logger.warning("Time out!")
            return False
        if not result:
            return False
        return result, qc
    
    
    def calculate_average_execution_time(self, device, num):
        source_path = str(helper.get_path_training_data()) + '/'
        folder_1 = Path(source_path + device + '_1')
        final_result = pd.DataFrame(columns=["quantum_circuit", "time_taken", "device"])
        for filename in folder_1.iterdir():
            if filename.suffix != ".csv":
                continue
            filename = str(filename).split("/")[-1]
            data = pd.read_csv(source_path + device + '_1' + '/' + filename)
            quantum_circuit = data['quantum_circuit']
            time_taken = data['time_taken']
            for i in range(2, num + 1):
                data = pd.read_csv(source_path + device + '_' + str(i) + '/' + filename)
                time_taken += data['time_taken']
            time_taken /= num
            final_result = pd.concat([final_result, pd.DataFrame({'quantum_circuit': quantum_circuit, 'time_taken': time_taken, 'device': device})], ignore_index=True)
        final_result.to_csv(source_path + device + '.csv', index=False)
    

    def calculate_average_execution_time_temp(self, device, num, list):
        source_path = str(helper.get_path_training_data()) + '/'
        time_taken = 0
        final_result = pd.DataFrame(columns=["quantum_circuit", "time_taken", "device"])
        for filename in list:
            for i in range(1, num + 1):
                data = pd.read_csv(source_path + device + '_' + str(i) + '/' + filename.split('.')[0] + '.csv')
                time_taken += data['time_taken']
            time_taken /= num
            final_result = pd.concat([final_result, pd.DataFrame({'quantum_circuit': filename, 'time_taken': time_taken, 'device': device})], ignore_index=True)
        final_result.to_csv(source_path + device + '.csv', index=False)
    

    def generate_training_sample_execution_time(self, data):
        training_data, scores_list = [], []
        for items in data.itertuples():
            circ = items[1]
            time_taken = items[2]
            device = items[3]
            qc = QuantumCircuit.from_qasm_str(circ)
            global_features = helper.create_feature_dict(qc)
            circ_graph_feature = circ_dag_converter.circ_to_dag_with_data(qc, device, list(global_features.values()), n_qubit=127)
            training_data.append(circ_graph_feature)
            scores_list.append(time_taken)
        return (training_data, scores_list)
