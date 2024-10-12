// Benchmark was created by MQT Bench on 2023-06-29
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: v1.0.0
// Qiskit version: {'qiskit-terra': '0.24.1', 'qiskit-aer': '0.12.0', 'qiskit-ignis': None, 'qiskit-ibmq-provider': '0.20.2', 'qiskit': '0.43.1', 'qiskit-nature': '0.6.2', 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.5.0', 'qiskit-machine-learning': '0.6.1'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[78];
creg meas[78];
ry(-pi/4) q[0];
ry(-0.9553166181245093) q[1];
ry(-pi/3) q[2];
ry(-1.1071487177940904) q[3];
ry(-1.1502619915109316) q[4];
ry(-1.183199640139716) q[5];
ry(-1.2094292028881888) q[6];
ry(-1.2309594173407747) q[7];
ry(-1.2490457723982544) q[8];
ry(-1.2645189576252271) q[9];
ry(-1.277953555066321) q[10];
ry(-1.2897614252920828) q[11];
ry(-1.3002465638163236) q[12];
ry(-1.3096389158918722) q[13];
ry(-1.318116071652818) q[14];
ry(-1.3258176636680323) q[15];
ry(-1.3328552019646882) q[16];
ry(-1.3393189628247182) q[17];
ry(-1.3452829208967654) q[18];
ry(-1.3508083493994372) q[19];
ry(-1.3559464937191843) q[20];
ry(-1.3607405877236576) q[21];
ry(-1.3652273956337226) q[22];
ry(-1.3694384060045657) q[23];
ry(-1.3734007669450157) q[24];
ry(-1.37713802635057) q[25];
ry(-1.38067072344843) q[26];
ry(-1.384016865713303) q[27];
ry(-1.387192316515978) q[28];
ry(-1.3902111126041985) q[29];
ry(-1.3930857259497849) q[30];
ry(-1.3958272811292076) q[31];
ry(-1.3984457368955736) q[32];
ry(-1.400950038711223) q[33];
ry(-1.4033482475752073) q[34];
ry(-1.4056476493802696) q[35];
ry(-1.4078548481843771) q[36];
ry(-1.409975846120432) q[37];
ry(-1.412016112149136) q[38];
ry(-1.4139806414504958) q[39];
ry(-1.4158740069240832) q[40];
ry(-1.417700404008042) q[41];
ry(-1.419463689817681) q[42];
ry(-1.4211674174353792) q[43];
ry(-1.4228148660461128) q[44];
ry(-1.4244090675006476) q[45];
ry(-1.4259528297963369) q[46];
ry(-1.4274487578895312) q[47];
ry(-1.4288992721907325) q[48];
ry(-1.4303066250413763) q[49];
ry(-1.431672915427498) q[50];
ry(-1.4330001021490115) q[51];
ry(-1.4342900156325915) q[52];
ry(-1.435544368550241) q[53];
ry(-1.4367647653836775) q[54];
ry(-1.4379527110560313) q[55];
ry(-1.4391096187364805) q[56];
ry(-1.4402368169098754) q[57];
ry(-1.441335555791786) q[58];
ry(-1.4424070131594149) q[59];
ry(-1.4434522996602146) q[60];
ry(-1.4444724636526147) q[61];
ry(-1.445468495626831) q[62];
ry(-1.446441332248135) q[63];
ry(-1.4473918600601101) q[64];
ry(-1.4483209188811768) q[65];
ry(-1.449229304923967) q[66];
ry(-1.4501177736638868) q[67];
ry(-1.4509870424803524) q[68];
ry(-1.4518377930916948) q[69];
ry(-1.4526706738025112) q[70];
ry(-1.4534863015803035) q[71];
ry(-1.4542852639765176) q[72];
ry(-1.4550681209055838) q[73];
ry(-1.45583540629419) q[74];
ry(-1.456587629611836) q[75];
ry(-1.457325277292633) q[76];
x q[77];
cz q[77],q[76];
ry(1.457325277292633) q[76];
cz q[76],q[75];
ry(1.456587629611836) q[75];
cz q[75],q[74];
ry(1.45583540629419) q[74];
cz q[74],q[73];
ry(1.4550681209055838) q[73];
cz q[73],q[72];
ry(1.4542852639765176) q[72];
cz q[72],q[71];
ry(1.4534863015803035) q[71];
cz q[71],q[70];
ry(1.4526706738025112) q[70];
cz q[70],q[69];
ry(1.4518377930916948) q[69];
cz q[69],q[68];
ry(1.4509870424803524) q[68];
cz q[68],q[67];
ry(1.4501177736638868) q[67];
cz q[67],q[66];
ry(1.449229304923967) q[66];
cz q[66],q[65];
ry(1.4483209188811768) q[65];
cz q[65],q[64];
ry(1.4473918600601101) q[64];
cz q[64],q[63];
ry(1.446441332248135) q[63];
cz q[63],q[62];
ry(1.445468495626831) q[62];
cz q[62],q[61];
ry(1.4444724636526147) q[61];
cz q[61],q[60];
ry(1.4434522996602146) q[60];
cz q[60],q[59];
ry(1.4424070131594149) q[59];
cz q[59],q[58];
ry(1.441335555791786) q[58];
cz q[58],q[57];
ry(1.4402368169098754) q[57];
cz q[57],q[56];
ry(1.4391096187364805) q[56];
cz q[56],q[55];
ry(1.4379527110560313) q[55];
cz q[55],q[54];
ry(1.4367647653836775) q[54];
cz q[54],q[53];
ry(1.435544368550241) q[53];
cz q[53],q[52];
ry(1.4342900156325915) q[52];
cz q[52],q[51];
ry(1.4330001021490115) q[51];
cz q[51],q[50];
ry(1.431672915427498) q[50];
cz q[50],q[49];
ry(1.4303066250413763) q[49];
cz q[49],q[48];
ry(1.4288992721907325) q[48];
cz q[48],q[47];
ry(1.4274487578895312) q[47];
cz q[47],q[46];
ry(1.4259528297963369) q[46];
cz q[46],q[45];
ry(1.4244090675006476) q[45];
cz q[45],q[44];
ry(1.4228148660461128) q[44];
cz q[44],q[43];
ry(1.4211674174353792) q[43];
cz q[43],q[42];
ry(1.419463689817681) q[42];
cz q[42],q[41];
ry(1.417700404008042) q[41];
cz q[41],q[40];
ry(1.4158740069240832) q[40];
cz q[40],q[39];
ry(1.4139806414504958) q[39];
cz q[39],q[38];
ry(1.412016112149136) q[38];
cz q[38],q[37];
ry(1.409975846120432) q[37];
cz q[37],q[36];
ry(1.4078548481843771) q[36];
cz q[36],q[35];
ry(1.4056476493802696) q[35];
cz q[35],q[34];
ry(1.4033482475752073) q[34];
cz q[34],q[33];
ry(1.400950038711223) q[33];
cz q[33],q[32];
ry(1.3984457368955736) q[32];
cz q[32],q[31];
ry(1.3958272811292076) q[31];
cz q[31],q[30];
ry(1.3930857259497849) q[30];
cz q[30],q[29];
ry(1.3902111126041985) q[29];
cz q[29],q[28];
ry(1.387192316515978) q[28];
cz q[28],q[27];
ry(1.384016865713303) q[27];
cz q[27],q[26];
ry(1.38067072344843) q[26];
cz q[26],q[25];
ry(1.37713802635057) q[25];
cz q[25],q[24];
ry(1.3734007669450157) q[24];
cz q[24],q[23];
ry(1.3694384060045657) q[23];
cz q[23],q[22];
ry(1.3652273956337226) q[22];
cz q[22],q[21];
ry(1.3607405877236576) q[21];
cz q[21],q[20];
ry(1.3559464937191843) q[20];
cz q[20],q[19];
ry(1.3508083493994372) q[19];
cz q[19],q[18];
ry(1.3452829208967654) q[18];
cz q[18],q[17];
ry(1.3393189628247182) q[17];
cz q[17],q[16];
ry(1.3328552019646882) q[16];
cz q[16],q[15];
ry(1.3258176636680323) q[15];
cz q[15],q[14];
ry(1.318116071652818) q[14];
cz q[14],q[13];
ry(1.3096389158918722) q[13];
cz q[13],q[12];
ry(1.3002465638163236) q[12];
cz q[12],q[11];
ry(1.2897614252920828) q[11];
cz q[11],q[10];
ry(1.277953555066321) q[10];
cz q[10],q[9];
cx q[76],q[77];
cx q[75],q[76];
cx q[74],q[75];
cx q[73],q[74];
cx q[72],q[73];
cx q[71],q[72];
cx q[70],q[71];
cx q[69],q[70];
cx q[68],q[69];
cx q[67],q[68];
cx q[66],q[67];
cx q[65],q[66];
cx q[64],q[65];
cx q[63],q[64];
cx q[62],q[63];
cx q[61],q[62];
cx q[60],q[61];
cx q[59],q[60];
cx q[58],q[59];
cx q[57],q[58];
cx q[56],q[57];
cx q[55],q[56];
cx q[54],q[55];
cx q[53],q[54];
cx q[52],q[53];
cx q[51],q[52];
cx q[50],q[51];
cx q[49],q[50];
cx q[48],q[49];
cx q[47],q[48];
cx q[46],q[47];
cx q[45],q[46];
cx q[44],q[45];
cx q[43],q[44];
cx q[42],q[43];
cx q[41],q[42];
cx q[40],q[41];
cx q[39],q[40];
cx q[38],q[39];
cx q[37],q[38];
cx q[36],q[37];
cx q[35],q[36];
cx q[34],q[35];
cx q[33],q[34];
cx q[32],q[33];
cx q[31],q[32];
cx q[30],q[31];
cx q[29],q[30];
cx q[28],q[29];
cx q[27],q[28];
cx q[26],q[27];
cx q[25],q[26];
cx q[24],q[25];
cx q[23],q[24];
cx q[22],q[23];
cx q[21],q[22];
cx q[20],q[21];
cx q[19],q[20];
cx q[18],q[19];
cx q[17],q[18];
cx q[16],q[17];
cx q[15],q[16];
cx q[14],q[15];
cx q[13],q[14];
cx q[12],q[13];
cx q[11],q[12];
cx q[10],q[11];
ry(1.2645189576252271) q[9];
cz q[9],q[8];
ry(1.2490457723982544) q[8];
cz q[8],q[7];
ry(1.2309594173407747) q[7];
cz q[7],q[6];
ry(1.2094292028881888) q[6];
cz q[6],q[5];
ry(1.183199640139716) q[5];
cz q[5],q[4];
ry(1.1502619915109316) q[4];
cz q[4],q[3];
ry(1.1071487177940904) q[3];
cz q[3],q[2];
ry(pi/3) q[2];
cz q[2],q[1];
ry(0.9553166181245093) q[1];
cz q[1],q[0];
ry(pi/4) q[0];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26],q[27],q[28],q[29],q[30],q[31],q[32],q[33],q[34],q[35],q[36],q[37],q[38],q[39],q[40],q[41],q[42],q[43],q[44],q[45],q[46],q[47],q[48],q[49],q[50],q[51],q[52],q[53],q[54],q[55],q[56],q[57],q[58],q[59],q[60],q[61],q[62],q[63],q[64],q[65],q[66],q[67],q[68],q[69],q[70],q[71],q[72],q[73],q[74],q[75],q[76],q[77];
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
measure q[12] -> meas[12];
measure q[13] -> meas[13];
measure q[14] -> meas[14];
measure q[15] -> meas[15];
measure q[16] -> meas[16];
measure q[17] -> meas[17];
measure q[18] -> meas[18];
measure q[19] -> meas[19];
measure q[20] -> meas[20];
measure q[21] -> meas[21];
measure q[22] -> meas[22];
measure q[23] -> meas[23];
measure q[24] -> meas[24];
measure q[25] -> meas[25];
measure q[26] -> meas[26];
measure q[27] -> meas[27];
measure q[28] -> meas[28];
measure q[29] -> meas[29];
measure q[30] -> meas[30];
measure q[31] -> meas[31];
measure q[32] -> meas[32];
measure q[33] -> meas[33];
measure q[34] -> meas[34];
measure q[35] -> meas[35];
measure q[36] -> meas[36];
measure q[37] -> meas[37];
measure q[38] -> meas[38];
measure q[39] -> meas[39];
measure q[40] -> meas[40];
measure q[41] -> meas[41];
measure q[42] -> meas[42];
measure q[43] -> meas[43];
measure q[44] -> meas[44];
measure q[45] -> meas[45];
measure q[46] -> meas[46];
measure q[47] -> meas[47];
measure q[48] -> meas[48];
measure q[49] -> meas[49];
measure q[50] -> meas[50];
measure q[51] -> meas[51];
measure q[52] -> meas[52];
measure q[53] -> meas[53];
measure q[54] -> meas[54];
measure q[55] -> meas[55];
measure q[56] -> meas[56];
measure q[57] -> meas[57];
measure q[58] -> meas[58];
measure q[59] -> meas[59];
measure q[60] -> meas[60];
measure q[61] -> meas[61];
measure q[62] -> meas[62];
measure q[63] -> meas[63];
measure q[64] -> meas[64];
measure q[65] -> meas[65];
measure q[66] -> meas[66];
measure q[67] -> meas[67];
measure q[68] -> meas[68];
measure q[69] -> meas[69];
measure q[70] -> meas[70];
measure q[71] -> meas[71];
measure q[72] -> meas[72];
measure q[73] -> meas[73];
measure q[74] -> meas[74];
measure q[75] -> meas[75];
measure q[76] -> meas[76];
measure q[77] -> meas[77];
