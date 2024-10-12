// Benchmark was created by MQT Bench on 2023-06-29
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: v1.0.0
// Qiskit version: {'qiskit-terra': '0.24.1', 'qiskit-aer': '0.12.0', 'qiskit-ignis': None, 'qiskit-ibmq-provider': '0.20.2', 'qiskit': '0.43.1', 'qiskit-nature': '0.6.2', 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.5.0', 'qiskit-machine-learning': '0.6.1'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg meas[12];
h q[0];
h q[1];
h q[2];
rzz(1.2894758534169126) q[1],q[2];
h q[3];
h q[4];
rzz(1.2894758534169126) q[3],q[4];
h q[5];
rzz(1.2894758534169126) q[1],q[5];
rx(-8.780070722797586) q[1];
h q[6];
rzz(1.2894758534169126) q[4],q[6];
rx(-8.780070722797586) q[4];
h q[7];
rzz(1.2894758534169126) q[5],q[7];
rx(-8.780070722797586) q[5];
rzz(1.2894758534169126) q[6],q[7];
rx(-8.780070722797586) q[6];
rx(-8.780070722797586) q[7];
h q[8];
rzz(1.2894758534169126) q[2],q[8];
rx(-8.780070722797586) q[2];
rzz(2.496850026336729) q[1],q[2];
rzz(2.496850026336729) q[1],q[5];
rx(-1.2894362949549918) q[1];
rzz(2.496850026336729) q[5],q[7];
rx(-1.2894362949549918) q[5];
h q[9];
rzz(1.2894758534169126) q[0],q[9];
h q[10];
rzz(1.2894758534169126) q[0],q[10];
rx(-8.780070722797586) q[0];
rzz(1.2894758534169126) q[3],q[10];
rx(-8.780070722797586) q[10];
rx(-8.780070722797586) q[3];
rzz(2.496850026336729) q[3],q[4];
rzz(2.496850026336729) q[4],q[6];
rx(-1.2894362949549918) q[4];
rzz(2.496850026336729) q[6],q[7];
rx(-1.2894362949549918) q[6];
rx(-1.2894362949549918) q[7];
h q[11];
rzz(1.2894758534169126) q[8],q[11];
rx(-8.780070722797586) q[8];
rzz(2.496850026336729) q[2],q[8];
rx(-1.2894362949549918) q[2];
rzz(1.2894758534169126) q[9],q[11];
rx(-8.780070722797586) q[11];
rzz(2.496850026336729) q[8],q[11];
rx(-1.2894362949549918) q[8];
rx(-8.780070722797586) q[9];
rzz(2.496850026336729) q[0],q[9];
rzz(2.496850026336729) q[0],q[10];
rx(-1.2894362949549918) q[0];
rzz(2.496850026336729) q[3],q[10];
rx(-1.2894362949549918) q[10];
rx(-1.2894362949549918) q[3];
rzz(2.496850026336729) q[9],q[11];
rx(-1.2894362949549918) q[11];
rx(-1.2894362949549918) q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
measure q[10] -> meas[10];
measure q[11] -> meas[11];
