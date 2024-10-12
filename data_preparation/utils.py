from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, cast
import logging
import signal
from typing import Any
logger = logging.getLogger("qet-predictor")

if TYPE_CHECKING:  # pragma: no cover
    from types import ModuleType

from importlib import import_module

import networkx as nx
import numpy as np
from pytket import __version__ as __tket_version__
from qiskit import QuantumCircuit, __qiskit_version__
from qiskit.providers.fake_provider import FakeMontreal, FakeWashington
from qiskit_optimization.applications import Maxcut

if TYPE_CHECKING or sys.version_info >= (3, 10, 0):  # pragma: no cover
    from importlib import metadata, resources
else:
    import importlib_metadata as metadata
    import importlib_resources as resources

if TYPE_CHECKING:  # pragma: no cover
    from qiskit.circuit import QuantumRegister, Qubit
    from qiskit_optimization import QuadraticProgram

from dataclasses import dataclass


@dataclass
class SupermarqFeatures:
    program_communication: float
    critical_depth: float
    entanglement_ratio: float
    parallelism: float
    liveness: float


def calc_qubit_index(qargs: list[Qubit], qregs: list[QuantumRegister], index: int) -> int:
    offset = 0
    for reg in qregs:
        if qargs[index] not in reg:
            offset += reg.size
        else:
            qubit_index: int = offset + reg.index(qargs[index])
            return qubit_index
    error_msg = f"Global qubit index for local qubit {index} index not found."
    raise ValueError(error_msg)


def calc_supermarq_features(
    qc: QuantumCircuit,
) -> SupermarqFeatures:
    connectivity_collection: list[list[int]] = []
    liveness_A_matrix = 0
    connectivity_collection = [[] for _ in range(qc.num_qubits)]

    for instruction, qargs, _ in qc.data:
        if instruction.name in ("barrier", "measure"):
            continue
        liveness_A_matrix += len(qargs)
        first_qubit = calc_qubit_index(qargs, qc.qregs, 0)
        all_indices = [first_qubit]
        if len(qargs) == 2:
            second_qubit = calc_qubit_index(qargs, qc.qregs, 1)
            all_indices.append(second_qubit)
        for qubit_index in all_indices:
            to_be_added_entries = all_indices.copy()
            to_be_added_entries.remove(int(qubit_index))
            connectivity_collection[int(qubit_index)].extend(to_be_added_entries)

    connectivity: list[int] = [len(set(connectivity_collection[i])) for i in range(qc.num_qubits)]

    count_ops = qc.count_ops()
    num_gates = sum(count_ops.values())
    # before subtracting the measure and barrier gates, check whether it is in the dict
    if "measure" in count_ops:
        num_gates -= count_ops.get("measure")
    if "barrier" in count_ops:
        num_gates -= count_ops.get("barrier")
    num_multiple_qubit_gates = qc.num_nonlocal_gates()
    depth = qc.depth(lambda x: x[0].name not in ("barrier", "measure"))
    program_communication = np.sum(connectivity) / (qc.num_qubits * (qc.num_qubits - 1))

    if num_multiple_qubit_gates == 0:
        critical_depth = 0.0
    else:
        critical_depth = (
            qc.depth(filter_function=lambda x: len(x[1]) > 1 and x[0].name != "barrier") / num_multiple_qubit_gates
        )

    entanglement_ratio = num_multiple_qubit_gates / num_gates
    assert num_multiple_qubit_gates <= num_gates

    parallelism = (num_gates / depth - 1) / (qc.num_qubits - 1)

    liveness = liveness_A_matrix / (depth * qc.num_qubits)

    assert 0 <= program_communication <= 1
    assert 0 <= critical_depth <= 1
    assert 0 <= entanglement_ratio <= 1
    assert 0 <= parallelism <= 1
    assert 0 <= liveness <= 1

    return SupermarqFeatures(
        program_communication,
        critical_depth,
        entanglement_ratio,
        parallelism,
        liveness,
    )


def timeout_watcher(func: Any, args: list[Any], timeout: int) -> Any:
    """Method that stops a function call after a given timeout limit."""

    class TimeoutException(Exception):  # Custom exception class
        pass

    def timeout_handler(_signum: Any, _frame: Any) -> None:  # Custom signal handler
        raise TimeoutException

    # Change the behavior of SIGALRM
    signal.signal(signal.SIGALRM, timeout_handler)

    signal.alarm(timeout)
    try:
        res = func(*args)
    except TimeoutException:
        logger.debug("Calculation/Generation exceeded timeout limit for " + func.__module__ + ", " + str(args[1:]))
        return False
    except Exception as e:
        logger.error("Something else went wrong: " + str(e))
        return False
    else:
        # Reset the alarm
        signal.alarm(0)

    return res