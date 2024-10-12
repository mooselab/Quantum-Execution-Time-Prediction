// Benchmark was created by MQT Bench on 2023-06-29
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: v1.0.0
// Qiskit version: {'qiskit-terra': '0.24.1', 'qiskit-aer': '0.12.0', 'qiskit-ignis': None, 'qiskit-ibmq-provider': '0.20.2', 'qiskit': '0.43.1', 'qiskit-nature': '0.6.2', 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.5.0', 'qiskit-machine-learning': '0.6.1'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg meas[5];
u2(pi/4,-pi) q[0];
u1(-1.490342548936602) q[1];
cx q[3],q[0];
tdg q[0];
cx q[2],q[0];
t q[0];
cx q[3],q[0];
u2(0,-3*pi/4) q[0];
rx(pi/2) q[3];
rzz(5.185719590928789) q[0],q[3];
u2(-pi/2,-pi) q[0];
rx(-pi/2) q[3];
u2(0,0) q[4];
cx q[4],q[1];
ry(-0.4689836211867622) q[1];
ry(-0.4689836211867622) q[4];
cx q[4],q[1];
u1(1.490342548936602) q[1];
cx q[2],q[1];
ccx q[1],q[3],q[0];
u2(-3.0295369846271756,-pi) q[0];
cz q[1],q[3];
p(0.8755706304852969) q[2];
h q[3];
cx q[1],q[3];
h q[3];
cu1(pi/2) q[1],q[3];
u3(1.9552569771057446,1.6644650681916158,-0.08695250949048283) q[1];
u1(pi/4) q[3];
u2(pi/2,-pi) q[4];
cu3(5.864966885210837,0.9110285368422897,4.684618627355123) q[2],q[4];
swap q[2],q[4];
cu3(0.9702813337800618,1.5693858735638762,5.465706745619813) q[2],q[0];
t q[0];
ry(3.772222268266399) q[4];
crx(4.07828399178064) q[4],q[2];
crx(2.3453197008450233) q[3],q[2];
z q[2];
cx q[2],q[1];
cx q[1],q[2];
h q[1];
rx(3.450765893010145) q[2];
u2(-1.8374614191958711,pi/2) q[3];
ch q[4],q[0];
h q[0];
cx q[4],q[0];
h q[0];
cu1(pi/2) q[4],q[0];
p(5.540732708314469) q[4];
cu3(5.432461787371337,0.43137627135395684,5.131693955927891) q[4],q[0];
cx q[0],q[1];
h q[1];
cu1(pi/2) q[0],q[1];
u3(1.2118830799977245,2.0936634444623183,1.7437382343344134) q[4];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
