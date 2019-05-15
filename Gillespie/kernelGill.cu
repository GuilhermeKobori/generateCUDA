#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

#define pow powf

#define SEED 23
#define cell 1.000000
#define ksynp53mRNA 0.001000
#define kdegp53mRNA 0.000100
#define ksynMdm2mRNA 0.000500
#define kdegMdm2mRNA 0.000500
#define ksynMdm2mRNAGSK3bp53 0.000700
#define ksynp53 0.007000
#define kdegp53 0.005000
#define kbinMdm2p53 0.001155
#define krelMdm2p53 0.000012
#define kbinGSK3bp53 0.000002
#define krelGSK3bp53 0.002000
#define ksynMdm2 0.000495
#define kdegMdm2 0.010000
#define kbinE1Ub 0.000200
#define kbinE2Ub 0.001000
#define kp53Ub 0.000050
#define kp53PolyUb 0.010000
#define kbinProt 0.000002
#define kactDUBp53 0.000000
#define kactDUBProtp53 0.000100
#define kactDUBMdm2 0.000000
#define kMdm2Ub 0.000005
#define kMdm2PUb 0.000007
#define kMdm2PolyUb 0.004560
#define kdam 0.080000
#define krepair 0.000020
#define kactATM 0.000100
#define kinactATM 0.000500
#define kphosp53 0.000200
#define kdephosp53 0.500000
#define kphosMdm2 2.000000
#define kdephosMdm2 0.500000
#define kphosMdm2GSK3b 0.005000
#define kphosMdm2GSK3bp53 0.500000
#define kphospTauGSK3bp53 0.100000
#define kphospTauGSK3b 0.000200
#define kdephospTau 0.010000
#define kbinMTTau 0.100000
#define krelMTTau 0.000100
#define ksynTau 0.000080
#define kbinTauProt 0.000000
#define kdegTau20SProt 0.010000
#define kaggTau 0.000000
#define kaggTauP1 0.000000
#define kaggTauP2 0.000000
#define ktangfor 0.001000
#define kinhibprot 0.000000
#define ksynp53mRNAAbeta 0.000010
#define kdamROS 0.000010
#define kgenROSAbeta 0.000020
#define kgenROSPlaque 0.000010
#define kgenROSGlia 0.000010
#define kproteff 1.000000
#define kremROS 0.000070
#define kprodAbeta 0.000019
#define kprodAbeta2 0.000019
#define kdegAbeta 0.000015
#define kaggAbeta 0.000003
#define kdisaggAbeta 0.000001
#define kdisaggAbeta1 0.000200
#define kdisaggAbeta2 0.000001
#define kdegAbetaGlia 0.005000
#define kpf 0.200000
#define kpg 0.150000
#define kpghalf 10.000000
#define kactglia1 0.000001
#define kactglia2 0.000001
#define kinactglia1 0.000005
#define kinactglia2 0.000005
#define kbinAbetaGlia 0.000010
#define krelAbetaGlia 0.000050
#define kdegAntiAb 0.000003
#define kbinAbantiAb 0.000001
#define speciesUpdate(i) \
switch (i) { \
case 0: \
p53_mRNA += 1.000000; \
break; \
case 1: \
p53_mRNA -= 1.000000; \
break; \
case 2: \
Mdm2_mRNA -= 1.000000; \
Mdm2_mRNA += 1.000000; \
Mdm2 += 1.000000; \
break; \
case 3: \
p53 -= 1.000000; \
p53 += 1.000000; \
Mdm2_mRNA += 1.000000; \
break; \
case 4: \
p53_P -= 1.000000; \
p53_P += 1.000000; \
Mdm2_mRNA += 1.000000; \
break; \
case 5: \
GSK3b_p53 -= 1.000000; \
GSK3b_p53 += 1.000000; \
Mdm2_mRNA += 1.000000; \
break; \
case 6: \
GSK3b_p53_P -= 1.000000; \
GSK3b_p53_P += 1.000000; \
Mdm2_mRNA += 1.000000; \
break; \
case 7: \
Mdm2_mRNA -= 1.000000; \
break; \
case 8: \
p53 -= 1.000000; \
Mdm2 -= 1.000000; \
Mdm2_p53 += 1.000000; \
break; \
case 9: \
Mdm2_p53 -= 1.000000; \
p53 += 1.000000; \
Mdm2 += 1.000000; \
break; \
case 10: \
GSK3b -= 1.000000; \
p53 -= 1.000000; \
GSK3b_p53 += 1.000000; \
break; \
case 11: \
GSK3b_p53 -= 1.000000; \
GSK3b += 1.000000; \
p53 += 1.000000; \
break; \
case 12: \
GSK3b -= 1.000000; \
p53_P -= 1.000000; \
GSK3b_p53_P += 1.000000; \
break; \
case 13: \
GSK3b_p53_P -= 1.000000; \
GSK3b += 1.000000; \
p53_P += 1.000000; \
break; \
case 14: \
E1 -= 1.000000; \
Ub -= 1.000000; \
E1_Ub += 1.000000; \
break; \
case 15: \
E2 -= 1.000000; \
E1_Ub -= 1.000000; \
E2_Ub += 1.000000; \
E1 += 1.000000; \
break; \
case 16: \
Mdm2 -= 1.000000; \
E2_Ub -= 1.000000; \
Mdm2_Ub += 1.000000; \
E2 += 1.000000; \
break; \
case 17: \
Mdm2_Ub -= 1.000000; \
E2_Ub -= 1.000000; \
Mdm2_Ub2 += 1.000000; \
E2 += 1.000000; \
break; \
case 18: \
Mdm2_Ub2 -= 1.000000; \
E2_Ub -= 1.000000; \
Mdm2_Ub3 += 1.000000; \
E2 += 1.000000; \
break; \
case 19: \
Mdm2_Ub3 -= 1.000000; \
E2_Ub -= 1.000000; \
Mdm2_Ub4 += 1.000000; \
E2 += 1.000000; \
break; \
case 20: \
Mdm2_Ub4 -= 1.000000; \
Mdm2DUB -= 1.000000; \
Mdm2_Ub3 += 1.000000; \
Mdm2DUB += 1.000000; \
Ub += 1.000000; \
break; \
case 21: \
Mdm2_Ub3 -= 1.000000; \
Mdm2DUB -= 1.000000; \
Mdm2_Ub2 += 1.000000; \
Mdm2DUB += 1.000000; \
Ub += 1.000000; \
break; \
case 22: \
Mdm2_Ub2 -= 1.000000; \
Mdm2DUB -= 1.000000; \
Mdm2_Ub += 1.000000; \
Mdm2DUB += 1.000000; \
Ub += 1.000000; \
break; \
case 23: \
Mdm2_Ub -= 1.000000; \
Mdm2DUB -= 1.000000; \
Mdm2 += 1.000000; \
Mdm2DUB += 1.000000; \
Ub += 1.000000; \
break; \
case 24: \
Mdm2_Ub4 -= 1.000000; \
Proteasome -= 1.000000; \
Mdm2_Ub4_Proteasome += 1.000000; \
break; \
case 25: \
Mdm2_Ub4_Proteasome -= 1.000000; \
Proteasome += 1.000000; \
Ub += 4.000000; \
break; \
case 26: \
p53_mRNA -= 1.000000; \
p53 += 1.000000; \
p53_mRNA += 1.000000; \
break; \
case 27: \
E2_Ub -= 1.000000; \
Mdm2_p53 -= 1.000000; \
Mdm2_p53_Ub += 1.000000; \
E2 += 1.000000; \
break; \
case 28: \
Mdm2_p53_Ub -= 1.000000; \
E2_Ub -= 1.000000; \
Mdm2_p53_Ub2 += 1.000000; \
E2 += 1.000000; \
break; \
case 29: \
Mdm2_p53_Ub2 -= 1.000000; \
E2_Ub -= 1.000000; \
Mdm2_p53_Ub3 += 1.000000; \
E2 += 1.000000; \
break; \
case 30: \
Mdm2_p53_Ub3 -= 1.000000; \
E2_Ub -= 1.000000; \
Mdm2_p53_Ub4 += 1.000000; \
E2 += 1.000000; \
break; \
case 31: \
Mdm2_p53_Ub4 -= 1.000000; \
p53DUB -= 1.000000; \
Mdm2_p53_Ub3 += 1.000000; \
p53DUB += 1.000000; \
Ub += 1.000000; \
break; \
case 32: \
Mdm2_p53_Ub3 -= 1.000000; \
p53DUB -= 1.000000; \
Mdm2_p53_Ub2 += 1.000000; \
p53DUB += 1.000000; \
Ub += 1.000000; \
break; \
case 33: \
Mdm2_p53_Ub2 -= 1.000000; \
p53DUB -= 1.000000; \
Mdm2_p53_Ub += 1.000000; \
p53DUB += 1.000000; \
Ub += 1.000000; \
break; \
case 34: \
Mdm2_p53_Ub -= 1.000000; \
p53DUB -= 1.000000; \
Mdm2_p53 += 1.000000; \
p53DUB += 1.000000; \
Ub += 1.000000; \
break; \
case 35: \
Mdm2_p53_Ub4 -= 1.000000; \
GSK3b -= 1.000000; \
Mdm2_P1_p53_Ub4 += 1.000000; \
GSK3b += 1.000000; \
break; \
case 36: \
Mdm2_p53_Ub4 -= 1.000000; \
GSK3b_p53 -= 1.000000; \
Mdm2_P1_p53_Ub4 += 1.000000; \
GSK3b_p53 += 1.000000; \
break; \
case 37: \
Mdm2_p53_Ub4 -= 1.000000; \
GSK3b_p53_P -= 1.000000; \
Mdm2_P1_p53_Ub4 += 1.000000; \
GSK3b_p53_P += 1.000000; \
break; \
case 38: \
Mdm2_P1_p53_Ub4 -= 1.000000; \
Proteasome -= 1.000000; \
p53_Ub4_Proteasome += 1.000000; \
Mdm2 += 1.000000; \
break; \
case 39: \
p53_Ub4_Proteasome -= 1.000000; \
Ub += 4.000000; \
Proteasome += 1.000000; \
break; \
case 40: \
Tau -= 1.000000; \
MT_Tau += 1.000000; \
break; \
case 41: \
MT_Tau -= 1.000000; \
Tau += 1.000000; \
break; \
case 42: \
GSK3b_p53 -= 1.000000; \
Tau -= 1.000000; \
GSK3b_p53 += 1.000000; \
Tau_P1 += 1.000000; \
break; \
case 43: \
GSK3b_p53 -= 1.000000; \
Tau_P1 -= 1.000000; \
GSK3b_p53 += 1.000000; \
Tau_P2 += 1.000000; \
break; \
case 44: \
GSK3b_p53_P -= 1.000000; \
Tau -= 1.000000; \
GSK3b_p53_P += 1.000000; \
Tau_P1 += 1.000000; \
break; \
case 45: \
GSK3b_p53_P -= 1.000000; \
Tau_P1 -= 1.000000; \
GSK3b_p53_P += 1.000000; \
Tau_P2 += 1.000000; \
break; \
case 46: \
GSK3b -= 1.000000; \
Tau -= 1.000000; \
GSK3b += 1.000000; \
Tau_P1 += 1.000000; \
break; \
case 47: \
GSK3b -= 1.000000; \
Tau_P1 -= 1.000000; \
GSK3b += 1.000000; \
Tau_P2 += 1.000000; \
break; \
case 48: \
Tau_P2 -= 1.000000; \
PP1 -= 1.000000; \
Tau_P1 += 1.000000; \
PP1 += 1.000000; \
break; \
case 49: \
Tau_P1 -= 1.000000; \
PP1 -= 1.000000; \
Tau += 1.000000; \
PP1 += 1.000000; \
break; \
case 50: \
Tau_P1 -= 2.000000; \
AggTau += 2.000000; \
break; \
case 51: \
Tau_P1 -= 1.000000; \
AggTau -= 1.000000; \
AggTau += 2.000000; \
break; \
case 52: \
Tau_P2 -= 2.000000; \
AggTau += 2.000000; \
break; \
case 53: \
Tau_P2 -= 1.000000; \
AggTau -= 1.000000; \
AggTau += 2.000000; \
break; \
case 54: \
Tau -= 2.000000; \
AggTau += 2.000000; \
break; \
case 55: \
Tau -= 1.000000; \
AggTau -= 1.000000; \
AggTau += 2.000000; \
break; \
case 56: \
AggTau -= 2.000000; \
NFT += 2.000000; \
break; \
case 57: \
AggTau -= 1.000000; \
NFT -= 1.000000; \
NFT += 2.000000; \
break; \
case 58: \
AggTau -= 1.000000; \
Proteasome -= 1.000000; \
AggTau_Proteasome += 1.000000; \
break; \
case 59: \
Abeta += 1.000000; \
break; \
case 60: \
GSK3b_p53 -= 1.000000; \
Abeta += 1.000000; \
GSK3b_p53 += 1.000000; \
break; \
case 61: \
GSK3b_p53_P -= 1.000000; \
Abeta += 1.000000; \
GSK3b_p53_P += 1.000000; \
break; \
case 62: \
AbetaDimer -= 1.000000; \
Proteasome -= 1.000000; \
AggAbeta_Proteasome += 1.000000; \
break; \
case 63: \
Abeta -= 1.000000; \
break; \
case 64: \
Abeta -= 1.000000; \
p53_mRNA += 1.000000; \
Abeta += 1.000000; \
break; \
case 65: \
IR -= 1.000000; \
IR += 1.000000; \
damDNA += 1.000000; \
break; \
case 66: \
damDNA -= 1.000000; \
break; \
case 67: \
damDNA -= 1.000000; \
ATMI -= 1.000000; \
damDNA += 1.000000; \
ATMA += 1.000000; \
break; \
case 68: \
p53 -= 1.000000; \
ATMA -= 1.000000; \
p53_P += 1.000000; \
ATMA += 1.000000; \
break; \
case 69: \
p53_P -= 1.000000; \
p53 += 1.000000; \
break; \
case 70: \
Mdm2 -= 1.000000; \
ATMA -= 1.000000; \
Mdm2_P += 1.000000; \
ATMA += 1.000000; \
break; \
case 71: \
Mdm2_P -= 1.000000; \
Mdm2 += 1.000000; \
break; \
case 72: \
Mdm2_P -= 1.000000; \
E2_Ub -= 1.000000; \
Mdm2_P_Ub += 1.000000; \
E2 += 1.000000; \
break; \
case 73: \
Mdm2_P_Ub -= 1.000000; \
E2_Ub -= 1.000000; \
Mdm2_P_Ub2 += 1.000000; \
E2 += 1.000000; \
break; \
case 74: \
Mdm2_P_Ub2 -= 1.000000; \
E2_Ub -= 1.000000; \
Mdm2_P_Ub3 += 1.000000; \
E2 += 1.000000; \
break; \
case 75: \
Mdm2_P_Ub3 -= 1.000000; \
E2_Ub -= 1.000000; \
Mdm2_P_Ub4 += 1.000000; \
E2 += 1.000000; \
break; \
case 76: \
Mdm2_P_Ub4 -= 1.000000; \
Mdm2DUB -= 1.000000; \
Mdm2_P_Ub3 += 1.000000; \
Mdm2DUB += 1.000000; \
Ub += 1.000000; \
break; \
case 77: \
Mdm2_P_Ub3 -= 1.000000; \
Mdm2DUB -= 1.000000; \
Mdm2_P_Ub2 += 1.000000; \
Mdm2DUB += 1.000000; \
Ub += 1.000000; \
break; \
case 78: \
Mdm2_P_Ub2 -= 1.000000; \
Mdm2DUB -= 1.000000; \
Mdm2_P_Ub += 1.000000; \
Mdm2DUB += 1.000000; \
Ub += 1.000000; \
break; \
case 79: \
Mdm2_P_Ub -= 1.000000; \
Mdm2DUB -= 1.000000; \
Mdm2_P += 1.000000; \
Mdm2DUB += 1.000000; \
Ub += 1.000000; \
break; \
case 80: \
Mdm2_P_Ub4 -= 1.000000; \
Proteasome -= 1.000000; \
Mdm2_P_Ub4_Proteasome += 1.000000; \
break; \
case 81: \
Mdm2_P_Ub4_Proteasome -= 1.000000; \
Proteasome += 1.000000; \
Ub += 4.000000; \
break; \
case 82: \
ATMA -= 1.000000; \
ATMI += 1.000000; \
break; \
case 83: \
Abeta -= 1.000000; \
Abeta += 1.000000; \
ROS += 1.000000; \
break; \
case 84: \
AbetaPlaque -= 1.000000; \
AbetaPlaque += 1.000000; \
ROS += 1.000000; \
break; \
case 85: \
AggAbeta_Proteasome -= 1.000000; \
AggAbeta_Proteasome += 1.000000; \
ROS += 1.000000; \
break; \
case 86: \
ROS -= 1.000000; \
ROS += 1.000000; \
damDNA += 1.000000; \
break; \
case 87: \
Tau += 1.000000; \
break; \
case 88: \
Tau -= 1.000000; \
Proteasome -= 1.000000; \
Proteasome_Tau += 1.000000; \
break; \
case 89: \
Proteasome_Tau -= 1.000000; \
Proteasome += 1.000000; \
break; \
case 90: \
Abeta -= 2.000000; \
AbetaDimer += 1.000000; \
break; \
case 91: \
AbetaDimer -= 2.000000; \
AbetaPlaque += 1.000000; \
break; \
case 92: \
AbetaDimer -= 1.000000; \
AbetaPlaque -= 1.000000; \
AbetaPlaque += 2.000000; \
break; \
case 93: \
AbetaDimer -= 1.000000; \
Abeta += 2.000000; \
break; \
case 94: \
AbetaPlaque -= 1.000000; \
AbetaDimer += 1.000000; \
disaggPlaque1 += 1.000000; \
break; \
case 95: \
AbetaPlaque -= 1.000000; \
antiAb -= 1.000000; \
AbetaDimer += 1.000000; \
antiAb += 1.000000; \
disaggPlaque2 += 1.000000; \
break; \
case 96: \
Abeta -= 1.000000; \
antiAb -= 1.000000; \
Abeta_antiAb += 1.000000; \
break; \
case 97: \
AbetaDimer -= 1.000000; \
antiAb -= 1.000000; \
AbetaDimer_antiAb += 1.000000; \
break; \
case 98: \
Abeta_antiAb -= 1.000000; \
antiAb += 1.000000; \
break; \
case 99: \
AbetaDimer_antiAb -= 1.000000; \
antiAb += 1.000000; \
break; \
case 100: \
GliaI -= 1.000000; \
AbetaPlaque -= 1.000000; \
GliaM1 += 1.000000; \
AbetaPlaque += 1.000000; \
break; \
case 101: \
GliaM1 -= 1.000000; \
AbetaPlaque -= 1.000000; \
GliaM2 += 1.000000; \
AbetaPlaque += 1.000000; \
break; \
case 102: \
GliaM2 -= 1.000000; \
antiAb -= 1.000000; \
GliaA += 1.000000; \
antiAb += 1.000000; \
break; \
case 103: \
GliaA -= 1.000000; \
GliaM2 += 1.000000; \
break; \
case 104: \
GliaM2 -= 1.000000; \
GliaM1 += 1.000000; \
break; \
case 105: \
GliaM1 -= 1.000000; \
GliaI += 1.000000; \
break; \
case 106: \
AbetaPlaque -= 1.000000; \
GliaA -= 1.000000; \
AbetaPlaque_GliaA += 1.000000; \
break; \
case 107: \
AbetaPlaque_GliaA -= 1.000000; \
AbetaPlaque += 1.000000; \
GliaA += 1.000000; \
break; \
case 108: \
AbetaPlaque_GliaA -= 1.000000; \
GliaA += 1.000000; \
degAbetaGlia += 1.000000; \
break; \
case 109: \
AbetaPlaque_GliaA -= 1.000000; \
AbetaPlaque_GliaA += 1.000000; \
ROS += 1.000000; \
break; \
case 110: \
antiAb -= 1.000000; \
break; \
case 111: \
ROS -= 1.000000; \
break; \
} \

__global__ 
void simulate (int numberOfExecutions, float* output, curandState *state, float step, float endTime, float segmentSize, float* Mdm2_aux, float* Mdm2_global, float* p53_aux, float* p53_global, float* Mdm2_p53_aux, float* Mdm2_p53_global, float* Mdm2_mRNA_aux, float* Mdm2_mRNA_global, float* p53_mRNA_aux, float* p53_mRNA_global, float* ATMA_aux, float* ATMA_global, float* ATMI_aux, float* ATMI_global, float* p53_P_aux, float* p53_P_global, float* Mdm2_P_aux, float* Mdm2_P_global, float* IR_aux, float* IR_global, float* ROS_aux, float* ROS_global, float* damDNA_aux, float* damDNA_global, float* E1_aux, float* E1_global, float* E2_aux, float* E2_global, float* E1_Ub_aux, float* E1_Ub_global, float* E2_Ub_aux, float* E2_Ub_global, float* Proteasome_aux, float* Proteasome_global, float* Ub_aux, float* Ub_global, float* p53DUB_aux, float* p53DUB_global, float* Mdm2DUB_aux, float* Mdm2DUB_global, float* DUB_aux, float* DUB_global, float* Mdm2_p53_Ub_aux, float* Mdm2_p53_Ub_global, float* Mdm2_p53_Ub2_aux, float* Mdm2_p53_Ub2_global, float* Mdm2_p53_Ub3_aux, float* Mdm2_p53_Ub3_global, float* Mdm2_p53_Ub4_aux, float* Mdm2_p53_Ub4_global, float* Mdm2_P1_p53_Ub4_aux, float* Mdm2_P1_p53_Ub4_global, float* Mdm2_Ub_aux, float* Mdm2_Ub_global, float* Mdm2_Ub2_aux, float* Mdm2_Ub2_global, float* Mdm2_Ub3_aux, float* Mdm2_Ub3_global, float* Mdm2_Ub4_aux, float* Mdm2_Ub4_global, float* Mdm2_P_Ub_aux, float* Mdm2_P_Ub_global, float* Mdm2_P_Ub2_aux, float* Mdm2_P_Ub2_global, float* Mdm2_P_Ub3_aux, float* Mdm2_P_Ub3_global, float* Mdm2_P_Ub4_aux, float* Mdm2_P_Ub4_global, float* p53_Ub4_Proteasome_aux, float* p53_Ub4_Proteasome_global, float* Mdm2_Ub4_Proteasome_aux, float* Mdm2_Ub4_Proteasome_global, float* Mdm2_P_Ub4_Proteasome_aux, float* Mdm2_P_Ub4_Proteasome_global, float* GSK3b_aux, float* GSK3b_global, float* GSK3b_p53_aux, float* GSK3b_p53_global, float* GSK3b_p53_P_aux, float* GSK3b_p53_P_global, float* Abeta_aux, float* Abeta_global, float* AggAbeta_Proteasome_aux, float* AggAbeta_Proteasome_global, float* AbetaPlaque_aux, float* AbetaPlaque_global, float* Tau_aux, float* Tau_global, float* Tau_P1_aux, float* Tau_P1_global, float* Tau_P2_aux, float* Tau_P2_global, float* MT_Tau_aux, float* MT_Tau_global, float* AggTau_aux, float* AggTau_global, float* AggTau_Proteasome_aux, float* AggTau_Proteasome_global, float* Proteasome_Tau_aux, float* Proteasome_Tau_global, float* PP1_aux, float* PP1_global, float* NFT_aux, float* NFT_global, float* ATP_aux, float* ATP_global, float* ADP_aux, float* ADP_global, float* AMP_aux, float* AMP_global, float* AbetaDimer_aux, float* AbetaDimer_global, float* AbetaPlaque_GliaA_aux, float* AbetaPlaque_GliaA_global, float* GliaI_aux, float* GliaI_global, float* GliaM1_aux, float* GliaM1_global, float* GliaM2_aux, float* GliaM2_global, float* GliaA_aux, float* GliaA_global, float* antiAb_aux, float* antiAb_global, float* Abeta_antiAb_aux, float* Abeta_antiAb_global, float* AbetaDimer_antiAb_aux, float* AbetaDimer_antiAb_global, float* degAbetaGlia_aux, float* degAbetaGlia_global, float* disaggPlaque1_aux, float* disaggPlaque1_global, float* disaggPlaque2_aux, float* disaggPlaque2_global, float* Source_aux, float* Source_global, float* Sink_aux, float* Sink_global) {
int reaction, stepCount = 0;
float time = numberOfExecutions*segmentSize;
float sum_p, sum_p_aux, timeStep, random;
float p[112];
int triggerEvent0 = 0;
if(time >= 345600) {triggerEvent0 = 1;}
float Mdm2;
if(numberOfExecutions == 0){
Mdm2 = *Mdm2_aux;
} else {
Mdm2 = Mdm2_global[threadIdx.x];
}
float p53;
if(numberOfExecutions == 0){
p53 = *p53_aux;
} else {
p53 = p53_global[threadIdx.x];
}
float Mdm2_p53;
if(numberOfExecutions == 0){
Mdm2_p53 = *Mdm2_p53_aux;
} else {
Mdm2_p53 = Mdm2_p53_global[threadIdx.x];
}
float Mdm2_mRNA;
if(numberOfExecutions == 0){
Mdm2_mRNA = *Mdm2_mRNA_aux;
} else {
Mdm2_mRNA = Mdm2_mRNA_global[threadIdx.x];
}
float p53_mRNA;
if(numberOfExecutions == 0){
p53_mRNA = *p53_mRNA_aux;
} else {
p53_mRNA = p53_mRNA_global[threadIdx.x];
}
float ATMA;
if(numberOfExecutions == 0){
ATMA = *ATMA_aux;
} else {
ATMA = ATMA_global[threadIdx.x];
}
float ATMI;
if(numberOfExecutions == 0){
ATMI = *ATMI_aux;
} else {
ATMI = ATMI_global[threadIdx.x];
}
float p53_P;
if(numberOfExecutions == 0){
p53_P = *p53_P_aux;
} else {
p53_P = p53_P_global[threadIdx.x];
}
float Mdm2_P;
if(numberOfExecutions == 0){
Mdm2_P = *Mdm2_P_aux;
} else {
Mdm2_P = Mdm2_P_global[threadIdx.x];
}
float IR;
if(numberOfExecutions == 0){
IR = *IR_aux;
} else {
IR = IR_global[threadIdx.x];
}
float ROS;
if(numberOfExecutions == 0){
ROS = *ROS_aux;
} else {
ROS = ROS_global[threadIdx.x];
}
float damDNA;
if(numberOfExecutions == 0){
damDNA = *damDNA_aux;
} else {
damDNA = damDNA_global[threadIdx.x];
}
float E1;
if(numberOfExecutions == 0){
E1 = *E1_aux;
} else {
E1 = E1_global[threadIdx.x];
}
float E2;
if(numberOfExecutions == 0){
E2 = *E2_aux;
} else {
E2 = E2_global[threadIdx.x];
}
float E1_Ub;
if(numberOfExecutions == 0){
E1_Ub = *E1_Ub_aux;
} else {
E1_Ub = E1_Ub_global[threadIdx.x];
}
float E2_Ub;
if(numberOfExecutions == 0){
E2_Ub = *E2_Ub_aux;
} else {
E2_Ub = E2_Ub_global[threadIdx.x];
}
float Proteasome;
if(numberOfExecutions == 0){
Proteasome = *Proteasome_aux;
} else {
Proteasome = Proteasome_global[threadIdx.x];
}
float Ub;
if(numberOfExecutions == 0){
Ub = *Ub_aux;
} else {
Ub = Ub_global[threadIdx.x];
}
float p53DUB;
if(numberOfExecutions == 0){
p53DUB = *p53DUB_aux;
} else {
p53DUB = p53DUB_global[threadIdx.x];
}
float Mdm2DUB;
if(numberOfExecutions == 0){
Mdm2DUB = *Mdm2DUB_aux;
} else {
Mdm2DUB = Mdm2DUB_global[threadIdx.x];
}
float DUB;
if(numberOfExecutions == 0){
DUB = *DUB_aux;
} else {
DUB = DUB_global[threadIdx.x];
}
float Mdm2_p53_Ub;
if(numberOfExecutions == 0){
Mdm2_p53_Ub = *Mdm2_p53_Ub_aux;
} else {
Mdm2_p53_Ub = Mdm2_p53_Ub_global[threadIdx.x];
}
float Mdm2_p53_Ub2;
if(numberOfExecutions == 0){
Mdm2_p53_Ub2 = *Mdm2_p53_Ub2_aux;
} else {
Mdm2_p53_Ub2 = Mdm2_p53_Ub2_global[threadIdx.x];
}
float Mdm2_p53_Ub3;
if(numberOfExecutions == 0){
Mdm2_p53_Ub3 = *Mdm2_p53_Ub3_aux;
} else {
Mdm2_p53_Ub3 = Mdm2_p53_Ub3_global[threadIdx.x];
}
float Mdm2_p53_Ub4;
if(numberOfExecutions == 0){
Mdm2_p53_Ub4 = *Mdm2_p53_Ub4_aux;
} else {
Mdm2_p53_Ub4 = Mdm2_p53_Ub4_global[threadIdx.x];
}
float Mdm2_P1_p53_Ub4;
if(numberOfExecutions == 0){
Mdm2_P1_p53_Ub4 = *Mdm2_P1_p53_Ub4_aux;
} else {
Mdm2_P1_p53_Ub4 = Mdm2_P1_p53_Ub4_global[threadIdx.x];
}
float Mdm2_Ub;
if(numberOfExecutions == 0){
Mdm2_Ub = *Mdm2_Ub_aux;
} else {
Mdm2_Ub = Mdm2_Ub_global[threadIdx.x];
}
float Mdm2_Ub2;
if(numberOfExecutions == 0){
Mdm2_Ub2 = *Mdm2_Ub2_aux;
} else {
Mdm2_Ub2 = Mdm2_Ub2_global[threadIdx.x];
}
float Mdm2_Ub3;
if(numberOfExecutions == 0){
Mdm2_Ub3 = *Mdm2_Ub3_aux;
} else {
Mdm2_Ub3 = Mdm2_Ub3_global[threadIdx.x];
}
float Mdm2_Ub4;
if(numberOfExecutions == 0){
Mdm2_Ub4 = *Mdm2_Ub4_aux;
} else {
Mdm2_Ub4 = Mdm2_Ub4_global[threadIdx.x];
}
float Mdm2_P_Ub;
if(numberOfExecutions == 0){
Mdm2_P_Ub = *Mdm2_P_Ub_aux;
} else {
Mdm2_P_Ub = Mdm2_P_Ub_global[threadIdx.x];
}
float Mdm2_P_Ub2;
if(numberOfExecutions == 0){
Mdm2_P_Ub2 = *Mdm2_P_Ub2_aux;
} else {
Mdm2_P_Ub2 = Mdm2_P_Ub2_global[threadIdx.x];
}
float Mdm2_P_Ub3;
if(numberOfExecutions == 0){
Mdm2_P_Ub3 = *Mdm2_P_Ub3_aux;
} else {
Mdm2_P_Ub3 = Mdm2_P_Ub3_global[threadIdx.x];
}
float Mdm2_P_Ub4;
if(numberOfExecutions == 0){
Mdm2_P_Ub4 = *Mdm2_P_Ub4_aux;
} else {
Mdm2_P_Ub4 = Mdm2_P_Ub4_global[threadIdx.x];
}
float p53_Ub4_Proteasome;
if(numberOfExecutions == 0){
p53_Ub4_Proteasome = *p53_Ub4_Proteasome_aux;
} else {
p53_Ub4_Proteasome = p53_Ub4_Proteasome_global[threadIdx.x];
}
float Mdm2_Ub4_Proteasome;
if(numberOfExecutions == 0){
Mdm2_Ub4_Proteasome = *Mdm2_Ub4_Proteasome_aux;
} else {
Mdm2_Ub4_Proteasome = Mdm2_Ub4_Proteasome_global[threadIdx.x];
}
float Mdm2_P_Ub4_Proteasome;
if(numberOfExecutions == 0){
Mdm2_P_Ub4_Proteasome = *Mdm2_P_Ub4_Proteasome_aux;
} else {
Mdm2_P_Ub4_Proteasome = Mdm2_P_Ub4_Proteasome_global[threadIdx.x];
}
float GSK3b;
if(numberOfExecutions == 0){
GSK3b = *GSK3b_aux;
} else {
GSK3b = GSK3b_global[threadIdx.x];
}
float GSK3b_p53;
if(numberOfExecutions == 0){
GSK3b_p53 = *GSK3b_p53_aux;
} else {
GSK3b_p53 = GSK3b_p53_global[threadIdx.x];
}
float GSK3b_p53_P;
if(numberOfExecutions == 0){
GSK3b_p53_P = *GSK3b_p53_P_aux;
} else {
GSK3b_p53_P = GSK3b_p53_P_global[threadIdx.x];
}
float Abeta;
if(numberOfExecutions == 0){
Abeta = *Abeta_aux;
} else {
Abeta = Abeta_global[threadIdx.x];
}
float AggAbeta_Proteasome;
if(numberOfExecutions == 0){
AggAbeta_Proteasome = *AggAbeta_Proteasome_aux;
} else {
AggAbeta_Proteasome = AggAbeta_Proteasome_global[threadIdx.x];
}
float AbetaPlaque;
if(numberOfExecutions == 0){
AbetaPlaque = *AbetaPlaque_aux;
} else {
AbetaPlaque = AbetaPlaque_global[threadIdx.x];
}
float Tau;
if(numberOfExecutions == 0){
Tau = *Tau_aux;
} else {
Tau = Tau_global[threadIdx.x];
}
float Tau_P1;
if(numberOfExecutions == 0){
Tau_P1 = *Tau_P1_aux;
} else {
Tau_P1 = Tau_P1_global[threadIdx.x];
}
float Tau_P2;
if(numberOfExecutions == 0){
Tau_P2 = *Tau_P2_aux;
} else {
Tau_P2 = Tau_P2_global[threadIdx.x];
}
float MT_Tau;
if(numberOfExecutions == 0){
MT_Tau = *MT_Tau_aux;
} else {
MT_Tau = MT_Tau_global[threadIdx.x];
}
float AggTau;
if(numberOfExecutions == 0){
AggTau = *AggTau_aux;
} else {
AggTau = AggTau_global[threadIdx.x];
}
float AggTau_Proteasome;
if(numberOfExecutions == 0){
AggTau_Proteasome = *AggTau_Proteasome_aux;
} else {
AggTau_Proteasome = AggTau_Proteasome_global[threadIdx.x];
}
float Proteasome_Tau;
if(numberOfExecutions == 0){
Proteasome_Tau = *Proteasome_Tau_aux;
} else {
Proteasome_Tau = Proteasome_Tau_global[threadIdx.x];
}
float PP1;
if(numberOfExecutions == 0){
PP1 = *PP1_aux;
} else {
PP1 = PP1_global[threadIdx.x];
}
float NFT;
if(numberOfExecutions == 0){
NFT = *NFT_aux;
} else {
NFT = NFT_global[threadIdx.x];
}
float ATP;
if(numberOfExecutions == 0){
ATP = *ATP_aux;
} else {
ATP = ATP_global[threadIdx.x];
}
float ADP;
if(numberOfExecutions == 0){
ADP = *ADP_aux;
} else {
ADP = ADP_global[threadIdx.x];
}
float AMP;
if(numberOfExecutions == 0){
AMP = *AMP_aux;
} else {
AMP = AMP_global[threadIdx.x];
}
float AbetaDimer;
if(numberOfExecutions == 0){
AbetaDimer = *AbetaDimer_aux;
} else {
AbetaDimer = AbetaDimer_global[threadIdx.x];
}
float AbetaPlaque_GliaA;
if(numberOfExecutions == 0){
AbetaPlaque_GliaA = *AbetaPlaque_GliaA_aux;
} else {
AbetaPlaque_GliaA = AbetaPlaque_GliaA_global[threadIdx.x];
}
float GliaI;
if(numberOfExecutions == 0){
GliaI = *GliaI_aux;
} else {
GliaI = GliaI_global[threadIdx.x];
}
float GliaM1;
if(numberOfExecutions == 0){
GliaM1 = *GliaM1_aux;
} else {
GliaM1 = GliaM1_global[threadIdx.x];
}
float GliaM2;
if(numberOfExecutions == 0){
GliaM2 = *GliaM2_aux;
} else {
GliaM2 = GliaM2_global[threadIdx.x];
}
float GliaA;
if(numberOfExecutions == 0){
GliaA = *GliaA_aux;
} else {
GliaA = GliaA_global[threadIdx.x];
}
float antiAb;
if(numberOfExecutions == 0){
antiAb = *antiAb_aux;
} else {
antiAb = antiAb_global[threadIdx.x];
}
float Abeta_antiAb;
if(numberOfExecutions == 0){
Abeta_antiAb = *Abeta_antiAb_aux;
} else {
Abeta_antiAb = Abeta_antiAb_global[threadIdx.x];
}
float AbetaDimer_antiAb;
if(numberOfExecutions == 0){
AbetaDimer_antiAb = *AbetaDimer_antiAb_aux;
} else {
AbetaDimer_antiAb = AbetaDimer_antiAb_global[threadIdx.x];
}
float degAbetaGlia;
if(numberOfExecutions == 0){
degAbetaGlia = *degAbetaGlia_aux;
} else {
degAbetaGlia = degAbetaGlia_global[threadIdx.x];
}
float disaggPlaque1;
if(numberOfExecutions == 0){
disaggPlaque1 = *disaggPlaque1_aux;
} else {
disaggPlaque1 = disaggPlaque1_global[threadIdx.x];
}
float disaggPlaque2;
if(numberOfExecutions == 0){
disaggPlaque2 = *disaggPlaque2_aux;
} else {
disaggPlaque2 = disaggPlaque2_global[threadIdx.x];
}
float Source;
if(numberOfExecutions == 0){
Source = *Source_aux;
} else {
Source = Source_global[threadIdx.x];
}
float Sink;
if(numberOfExecutions == 0){
Sink = *Sink_aux;
} else {
Sink = Sink_global[threadIdx.x];
}
while(time < endTime && time < (numberOfExecutions + 1)*segmentSize){
p[0] = ksynp53mRNA * Source; 
p[1] = kdegp53mRNA * p53_mRNA; 
p[2] = ksynMdm2 * Mdm2_mRNA; 
p[3] = ksynMdm2mRNA * p53; 
p[4] = ksynMdm2mRNA * p53_P; 
p[5] = ksynMdm2mRNAGSK3bp53 * GSK3b_p53; 
p[6] = ksynMdm2mRNAGSK3bp53 * GSK3b_p53_P; 
p[7] = kdegMdm2mRNA * Mdm2_mRNA; 
p[8] = kbinMdm2p53 * p53 * Mdm2; 
p[9] = krelMdm2p53 * Mdm2_p53; 
p[10] = kbinGSK3bp53 * GSK3b * p53; 
p[11] = krelGSK3bp53 * GSK3b_p53; 
p[12] = kbinGSK3bp53 * GSK3b * p53_P; 
p[13] = krelGSK3bp53 * GSK3b_p53_P; 
p[14] = kbinE1Ub * E1 * Ub * ATP / (5000 + ATP); 
p[15] = kbinE2Ub * E2 * E1_Ub; 
p[16] = kMdm2Ub * Mdm2 * E2_Ub; 
p[17] = kMdm2PolyUb * Mdm2_Ub * E2_Ub; 
p[18] = kMdm2PolyUb * Mdm2_Ub2 * E2_Ub; 
p[19] = kMdm2PolyUb * Mdm2_Ub3 * E2_Ub; 
p[20] = kactDUBMdm2 * Mdm2_Ub4 * Mdm2DUB; 
p[21] = kactDUBMdm2 * Mdm2_Ub3 * Mdm2DUB; 
p[22] = kactDUBMdm2 * Mdm2_Ub2 * Mdm2DUB; 
p[23] = kactDUBMdm2 * Mdm2_Ub * Mdm2DUB; 
p[24] = kbinProt * Mdm2_Ub4 * Proteasome; 
p[25] = kdegMdm2 * Mdm2_Ub4_Proteasome * kproteff; 
p[26] = ksynp53 * p53_mRNA; 
p[27] = kp53Ub * E2_Ub * Mdm2_p53; 
p[28] = kp53PolyUb * Mdm2_p53_Ub * E2_Ub; 
p[29] = kp53PolyUb * Mdm2_p53_Ub2 * E2_Ub; 
p[30] = kp53PolyUb * Mdm2_p53_Ub3 * E2_Ub; 
p[31] = kactDUBp53 * Mdm2_p53_Ub4 * p53DUB; 
p[32] = kactDUBp53 * Mdm2_p53_Ub3 * p53DUB; 
p[33] = kactDUBp53 * Mdm2_p53_Ub2 * p53DUB; 
p[34] = kactDUBp53 * Mdm2_p53_Ub * p53DUB; 
p[35] = kphosMdm2GSK3b * Mdm2_p53_Ub4 * GSK3b; 
p[36] = kphosMdm2GSK3bp53 * Mdm2_p53_Ub4 * GSK3b_p53; 
p[37] = kphosMdm2GSK3bp53 * Mdm2_p53_Ub4 * GSK3b_p53_P; 
p[38] = kbinProt * Mdm2_P1_p53_Ub4 * Proteasome; 
p[39] = kdegp53 * kproteff * p53_Ub4_Proteasome * ATP / (5000 + ATP); 
p[40] = kbinMTTau * Tau; 
p[41] = krelMTTau * MT_Tau; 
p[42] = kphospTauGSK3bp53 * GSK3b_p53 * Tau; 
p[43] = kphospTauGSK3bp53 * GSK3b_p53 * Tau_P1; 
p[44] = kphospTauGSK3bp53 * GSK3b_p53_P * Tau; 
p[45] = kphospTauGSK3bp53 * GSK3b_p53_P * Tau_P1; 
p[46] = kphospTauGSK3b * GSK3b * Tau; 
p[47] = kphospTauGSK3b * GSK3b * Tau_P1; 
p[48] = kdephospTau * Tau_P2 * PP1; 
p[49] = kdephospTau * Tau_P1 * PP1; 
p[50] = kaggTauP1 * Tau_P1 * (Tau_P1 - 1) * 0.5; 
p[51] = kaggTauP1 * Tau_P1 * AggTau; 
p[52] = kaggTauP2 * Tau_P2 * (Tau_P2 - 1) * 0.5; 
p[53] = kaggTauP2 * Tau_P2 * AggTau; 
p[54] = kaggTau * Tau * (Tau - 1) * 0.5; 
p[55] = kaggTau * Tau * AggTau; 
p[56] = ktangfor * AggTau * (AggTau - 1) * 0.5; 
p[57] = ktangfor * AggTau * NFT; 
p[58] = kinhibprot * AggTau * Proteasome; 
p[59] = kprodAbeta * Source; 
p[60] = kprodAbeta2 * GSK3b_p53; 
p[61] = kprodAbeta2 * GSK3b_p53_P; 
p[62] = kinhibprot * AbetaDimer * Proteasome; 
p[63] = kdegAbeta * Abeta; 
p[64] = ksynp53mRNAAbeta * Abeta; 
p[65] = kdam * IR; 
p[66] = krepair * damDNA; 
p[67] = kactATM * damDNA * ATMI; 
p[68] = kphosp53 * p53 * ATMA; 
p[69] = kdephosp53 * p53_P; 
p[70] = kphosMdm2 * Mdm2 * ATMA; 
p[71] = kdephosMdm2 * Mdm2_P; 
p[72] = kMdm2PUb * Mdm2_P * E2_Ub; 
p[73] = kMdm2PolyUb * Mdm2_P_Ub * E2_Ub; 
p[74] = kMdm2PolyUb * Mdm2_P_Ub2 * E2_Ub; 
p[75] = kMdm2PolyUb * Mdm2_P_Ub3 * E2_Ub; 
p[76] = kactDUBMdm2 * Mdm2_P_Ub4 * Mdm2DUB; 
p[77] = kactDUBMdm2 * Mdm2_P_Ub3 * Mdm2DUB; 
p[78] = kactDUBMdm2 * Mdm2_P_Ub2 * Mdm2DUB; 
p[79] = kactDUBMdm2 * Mdm2_P_Ub * Mdm2DUB; 
p[80] = kbinProt * Mdm2_P_Ub4 * Proteasome; 
p[81] = kdegMdm2 * Mdm2_P_Ub4_Proteasome * kproteff; 
p[82] = kinactATM * ATMA; 
p[83] = kgenROSAbeta * Abeta; 
p[84] = kgenROSPlaque * AbetaPlaque; 
p[85] = kgenROSAbeta * AggAbeta_Proteasome; 
p[86] = kdamROS * ROS; 
p[87] = ksynTau * Source; 
p[88] = kbinTauProt * Tau * Proteasome; 
p[89] = kdegTau20SProt * Proteasome_Tau; 
p[90] = kaggAbeta * Abeta * (Abeta - 1) * 0.5; 
p[91] = kpf * AbetaDimer * (AbetaDimer - 1) * 0.5; 
p[92] = kpg * AbetaDimer * pow(AbetaPlaque, 2) / (pow(kpghalf, 2) + pow(AbetaPlaque, 2)); 
p[93] = kdisaggAbeta * AbetaDimer; 
p[94] = kdisaggAbeta1 * AbetaPlaque; 
p[95] = kdisaggAbeta2 * antiAb * AbetaPlaque; 
p[96] = kbinAbantiAb * Abeta * antiAb; 
p[97] = kbinAbantiAb * AbetaDimer * antiAb; 
p[98] = 10 * kdegAbeta * Abeta_antiAb; 
p[99] = 10 * kdegAbeta * AbetaDimer_antiAb; 
p[100] = kactglia1 * GliaI * AbetaPlaque; 
p[101] = kactglia1 * GliaM1 * AbetaPlaque; 
p[102] = kactglia2 * GliaM2 * antiAb; 
p[103] = kinactglia1 * GliaA; 
p[104] = kinactglia2 * GliaM2; 
p[105] = kinactglia2 * GliaM1; 
p[106] = kbinAbetaGlia * AbetaPlaque * GliaA; 
p[107] = krelAbetaGlia * AbetaPlaque_GliaA; 
p[108] = kdegAbetaGlia * AbetaPlaque_GliaA; 
p[109] = kgenROSGlia * AbetaPlaque_GliaA; 
p[110] = kdegAntiAb * antiAb; 
p[111] = kremROS * ROS; 
if(time >= step * stepCount){
atomicAdd(&output[stepCount*69 + 0], Mdm2);
atomicAdd(&output[stepCount*69 + 1], p53);
atomicAdd(&output[stepCount*69 + 2], Mdm2_p53);
atomicAdd(&output[stepCount*69 + 3], Mdm2_mRNA);
atomicAdd(&output[stepCount*69 + 4], p53_mRNA);
atomicAdd(&output[stepCount*69 + 5], ATMA);
atomicAdd(&output[stepCount*69 + 6], ATMI);
atomicAdd(&output[stepCount*69 + 7], p53_P);
atomicAdd(&output[stepCount*69 + 8], Mdm2_P);
atomicAdd(&output[stepCount*69 + 9], IR);
atomicAdd(&output[stepCount*69 + 10], ROS);
atomicAdd(&output[stepCount*69 + 11], damDNA);
atomicAdd(&output[stepCount*69 + 12], E1);
atomicAdd(&output[stepCount*69 + 13], E2);
atomicAdd(&output[stepCount*69 + 14], E1_Ub);
atomicAdd(&output[stepCount*69 + 15], E2_Ub);
atomicAdd(&output[stepCount*69 + 16], Proteasome);
atomicAdd(&output[stepCount*69 + 17], Ub);
atomicAdd(&output[stepCount*69 + 18], p53DUB);
atomicAdd(&output[stepCount*69 + 19], Mdm2DUB);
atomicAdd(&output[stepCount*69 + 20], DUB);
atomicAdd(&output[stepCount*69 + 21], Mdm2_p53_Ub);
atomicAdd(&output[stepCount*69 + 22], Mdm2_p53_Ub2);
atomicAdd(&output[stepCount*69 + 23], Mdm2_p53_Ub3);
atomicAdd(&output[stepCount*69 + 24], Mdm2_p53_Ub4);
atomicAdd(&output[stepCount*69 + 25], Mdm2_P1_p53_Ub4);
atomicAdd(&output[stepCount*69 + 26], Mdm2_Ub);
atomicAdd(&output[stepCount*69 + 27], Mdm2_Ub2);
atomicAdd(&output[stepCount*69 + 28], Mdm2_Ub3);
atomicAdd(&output[stepCount*69 + 29], Mdm2_Ub4);
atomicAdd(&output[stepCount*69 + 30], Mdm2_P_Ub);
atomicAdd(&output[stepCount*69 + 31], Mdm2_P_Ub2);
atomicAdd(&output[stepCount*69 + 32], Mdm2_P_Ub3);
atomicAdd(&output[stepCount*69 + 33], Mdm2_P_Ub4);
atomicAdd(&output[stepCount*69 + 34], p53_Ub4_Proteasome);
atomicAdd(&output[stepCount*69 + 35], Mdm2_Ub4_Proteasome);
atomicAdd(&output[stepCount*69 + 36], Mdm2_P_Ub4_Proteasome);
atomicAdd(&output[stepCount*69 + 37], GSK3b);
atomicAdd(&output[stepCount*69 + 38], GSK3b_p53);
atomicAdd(&output[stepCount*69 + 39], GSK3b_p53_P);
atomicAdd(&output[stepCount*69 + 40], Abeta);
atomicAdd(&output[stepCount*69 + 41], AggAbeta_Proteasome);
atomicAdd(&output[stepCount*69 + 42], AbetaPlaque);
atomicAdd(&output[stepCount*69 + 43], Tau);
atomicAdd(&output[stepCount*69 + 44], Tau_P1);
atomicAdd(&output[stepCount*69 + 45], Tau_P2);
atomicAdd(&output[stepCount*69 + 46], MT_Tau);
atomicAdd(&output[stepCount*69 + 47], AggTau);
atomicAdd(&output[stepCount*69 + 48], AggTau_Proteasome);
atomicAdd(&output[stepCount*69 + 49], Proteasome_Tau);
atomicAdd(&output[stepCount*69 + 50], PP1);
atomicAdd(&output[stepCount*69 + 51], NFT);
atomicAdd(&output[stepCount*69 + 52], ATP);
atomicAdd(&output[stepCount*69 + 53], ADP);
atomicAdd(&output[stepCount*69 + 54], AMP);
atomicAdd(&output[stepCount*69 + 55], AbetaDimer);
atomicAdd(&output[stepCount*69 + 56], AbetaPlaque_GliaA);
atomicAdd(&output[stepCount*69 + 57], GliaI);
atomicAdd(&output[stepCount*69 + 58], GliaM1);
atomicAdd(&output[stepCount*69 + 59], GliaM2);
atomicAdd(&output[stepCount*69 + 60], GliaA);
atomicAdd(&output[stepCount*69 + 61], antiAb);
atomicAdd(&output[stepCount*69 + 62], Abeta_antiAb);
atomicAdd(&output[stepCount*69 + 63], AbetaDimer_antiAb);
atomicAdd(&output[stepCount*69 + 64], degAbetaGlia);
atomicAdd(&output[stepCount*69 + 65], disaggPlaque1);
atomicAdd(&output[stepCount*69 + 66], disaggPlaque2);
atomicAdd(&output[stepCount*69 + 67], Source);
atomicAdd(&output[stepCount*69 + 68], Sink);
stepCount++;
}
sum_p = 0;
for(int i = 0; i < 112; i++){
sum_p += p[i];
}
curandState localState = state[threadIdx.x];
random = curand_uniform(&localState);
if(sum_p > 0) timeStep = -log(random)/sum_p;
else break;
random = curand_uniform(&localState);
reaction = -1;
sum_p_aux = 0;
random *= sum_p;
for(int i = 0; i < 112; i++){
sum_p_aux += p[i];
if(random < sum_p_aux){
reaction = i;
break;
}
}
speciesUpdate(reaction);
if(triggerEvent0 == 0 && time >= 345600){
triggerEvent0 = 1;
antiAb += 50;
}
time += timeStep;
}
Mdm2_global[threadIdx.x] = Mdm2;
p53_global[threadIdx.x] = p53;
Mdm2_p53_global[threadIdx.x] = Mdm2_p53;
Mdm2_mRNA_global[threadIdx.x] = Mdm2_mRNA;
p53_mRNA_global[threadIdx.x] = p53_mRNA;
ATMA_global[threadIdx.x] = ATMA;
ATMI_global[threadIdx.x] = ATMI;
p53_P_global[threadIdx.x] = p53_P;
Mdm2_P_global[threadIdx.x] = Mdm2_P;
IR_global[threadIdx.x] = IR;
ROS_global[threadIdx.x] = ROS;
damDNA_global[threadIdx.x] = damDNA;
E1_global[threadIdx.x] = E1;
E2_global[threadIdx.x] = E2;
E1_Ub_global[threadIdx.x] = E1_Ub;
E2_Ub_global[threadIdx.x] = E2_Ub;
Proteasome_global[threadIdx.x] = Proteasome;
Ub_global[threadIdx.x] = Ub;
p53DUB_global[threadIdx.x] = p53DUB;
Mdm2DUB_global[threadIdx.x] = Mdm2DUB;
DUB_global[threadIdx.x] = DUB;
Mdm2_p53_Ub_global[threadIdx.x] = Mdm2_p53_Ub;
Mdm2_p53_Ub2_global[threadIdx.x] = Mdm2_p53_Ub2;
Mdm2_p53_Ub3_global[threadIdx.x] = Mdm2_p53_Ub3;
Mdm2_p53_Ub4_global[threadIdx.x] = Mdm2_p53_Ub4;
Mdm2_P1_p53_Ub4_global[threadIdx.x] = Mdm2_P1_p53_Ub4;
Mdm2_Ub_global[threadIdx.x] = Mdm2_Ub;
Mdm2_Ub2_global[threadIdx.x] = Mdm2_Ub2;
Mdm2_Ub3_global[threadIdx.x] = Mdm2_Ub3;
Mdm2_Ub4_global[threadIdx.x] = Mdm2_Ub4;
Mdm2_P_Ub_global[threadIdx.x] = Mdm2_P_Ub;
Mdm2_P_Ub2_global[threadIdx.x] = Mdm2_P_Ub2;
Mdm2_P_Ub3_global[threadIdx.x] = Mdm2_P_Ub3;
Mdm2_P_Ub4_global[threadIdx.x] = Mdm2_P_Ub4;
p53_Ub4_Proteasome_global[threadIdx.x] = p53_Ub4_Proteasome;
Mdm2_Ub4_Proteasome_global[threadIdx.x] = Mdm2_Ub4_Proteasome;
Mdm2_P_Ub4_Proteasome_global[threadIdx.x] = Mdm2_P_Ub4_Proteasome;
GSK3b_global[threadIdx.x] = GSK3b;
GSK3b_p53_global[threadIdx.x] = GSK3b_p53;
GSK3b_p53_P_global[threadIdx.x] = GSK3b_p53_P;
Abeta_global[threadIdx.x] = Abeta;
AggAbeta_Proteasome_global[threadIdx.x] = AggAbeta_Proteasome;
AbetaPlaque_global[threadIdx.x] = AbetaPlaque;
Tau_global[threadIdx.x] = Tau;
Tau_P1_global[threadIdx.x] = Tau_P1;
Tau_P2_global[threadIdx.x] = Tau_P2;
MT_Tau_global[threadIdx.x] = MT_Tau;
AggTau_global[threadIdx.x] = AggTau;
AggTau_Proteasome_global[threadIdx.x] = AggTau_Proteasome;
Proteasome_Tau_global[threadIdx.x] = Proteasome_Tau;
PP1_global[threadIdx.x] = PP1;
NFT_global[threadIdx.x] = NFT;
ATP_global[threadIdx.x] = ATP;
ADP_global[threadIdx.x] = ADP;
AMP_global[threadIdx.x] = AMP;
AbetaDimer_global[threadIdx.x] = AbetaDimer;
AbetaPlaque_GliaA_global[threadIdx.x] = AbetaPlaque_GliaA;
GliaI_global[threadIdx.x] = GliaI;
GliaM1_global[threadIdx.x] = GliaM1;
GliaM2_global[threadIdx.x] = GliaM2;
GliaA_global[threadIdx.x] = GliaA;
antiAb_global[threadIdx.x] = antiAb;
Abeta_antiAb_global[threadIdx.x] = Abeta_antiAb;
AbetaDimer_antiAb_global[threadIdx.x] = AbetaDimer_antiAb;
degAbetaGlia_global[threadIdx.x] = degAbetaGlia;
disaggPlaque1_global[threadIdx.x] = disaggPlaque1;
disaggPlaque2_global[threadIdx.x] = disaggPlaque2;
Source_global[threadIdx.x] = Source;
Sink_global[threadIdx.x] = Sink;
}

__global__ 
void initCurand(curandState* state, unsigned long long seed){
curand_init(seed, threadIdx.x, 0, &state[threadIdx.x]);
}

int main()
{
cudaError_t cudaStatus;
float* output;
float* dev_output;
output = (float*)malloc(34*69*sizeof(float));
for(int i = 0; i < 34*69; i++){
output[i] = 0;
}
cudaStatus = cudaMalloc(&dev_output, 34*69*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_output, output, 34*69*sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2 = 5.000000;
float* dev_Mdm2 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2, &Mdm2, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2_global;
cudaStatus = cudaMalloc(&Mdm2_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float p53 = 5.000000;
float* dev_p53 = 0;
cudaStatus = cudaMalloc(&dev_p53, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_p53, &p53, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* p53_global;
cudaStatus = cudaMalloc(&p53_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Mdm2_p53 = 95.000000;
float* dev_Mdm2_p53 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_p53, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_p53, &Mdm2_p53, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2_p53_global;
cudaStatus = cudaMalloc(&Mdm2_p53_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Mdm2_mRNA = 10.000000;
float* dev_Mdm2_mRNA = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_mRNA, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_mRNA, &Mdm2_mRNA, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2_mRNA_global;
cudaStatus = cudaMalloc(&Mdm2_mRNA_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float p53_mRNA = 10.000000;
float* dev_p53_mRNA = 0;
cudaStatus = cudaMalloc(&dev_p53_mRNA, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_p53_mRNA, &p53_mRNA, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* p53_mRNA_global;
cudaStatus = cudaMalloc(&p53_mRNA_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float ATMA = 0.000000;
float* dev_ATMA = 0;
cudaStatus = cudaMalloc(&dev_ATMA, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_ATMA, &ATMA, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* ATMA_global;
cudaStatus = cudaMalloc(&ATMA_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float ATMI = 200.000000;
float* dev_ATMI = 0;
cudaStatus = cudaMalloc(&dev_ATMI, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_ATMI, &ATMI, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* ATMI_global;
cudaStatus = cudaMalloc(&ATMI_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float p53_P = 0.000000;
float* dev_p53_P = 0;
cudaStatus = cudaMalloc(&dev_p53_P, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_p53_P, &p53_P, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* p53_P_global;
cudaStatus = cudaMalloc(&p53_P_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Mdm2_P = 0.000000;
float* dev_Mdm2_P = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_P, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_P, &Mdm2_P, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2_P_global;
cudaStatus = cudaMalloc(&Mdm2_P_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float IR = 0.000000;
float* dev_IR = 0;
cudaStatus = cudaMalloc(&dev_IR, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_IR, &IR, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* IR_global;
cudaStatus = cudaMalloc(&IR_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float ROS = 0.000000;
float* dev_ROS = 0;
cudaStatus = cudaMalloc(&dev_ROS, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_ROS, &ROS, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* ROS_global;
cudaStatus = cudaMalloc(&ROS_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float damDNA = 0.000000;
float* dev_damDNA = 0;
cudaStatus = cudaMalloc(&dev_damDNA, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_damDNA, &damDNA, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* damDNA_global;
cudaStatus = cudaMalloc(&damDNA_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float E1 = 100.000000;
float* dev_E1 = 0;
cudaStatus = cudaMalloc(&dev_E1, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_E1, &E1, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* E1_global;
cudaStatus = cudaMalloc(&E1_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float E2 = 100.000000;
float* dev_E2 = 0;
cudaStatus = cudaMalloc(&dev_E2, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_E2, &E2, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* E2_global;
cudaStatus = cudaMalloc(&E2_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float E1_Ub = 0.000000;
float* dev_E1_Ub = 0;
cudaStatus = cudaMalloc(&dev_E1_Ub, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_E1_Ub, &E1_Ub, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* E1_Ub_global;
cudaStatus = cudaMalloc(&E1_Ub_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float E2_Ub = 0.000000;
float* dev_E2_Ub = 0;
cudaStatus = cudaMalloc(&dev_E2_Ub, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_E2_Ub, &E2_Ub, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* E2_Ub_global;
cudaStatus = cudaMalloc(&E2_Ub_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Proteasome = 500.000000;
float* dev_Proteasome = 0;
cudaStatus = cudaMalloc(&dev_Proteasome, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Proteasome, &Proteasome, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Proteasome_global;
cudaStatus = cudaMalloc(&Proteasome_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Ub = 4000.000000;
float* dev_Ub = 0;
cudaStatus = cudaMalloc(&dev_Ub, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Ub, &Ub, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Ub_global;
cudaStatus = cudaMalloc(&Ub_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float p53DUB = 200.000000;
float* dev_p53DUB = 0;
cudaStatus = cudaMalloc(&dev_p53DUB, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_p53DUB, &p53DUB, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* p53DUB_global;
cudaStatus = cudaMalloc(&p53DUB_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Mdm2DUB = 200.000000;
float* dev_Mdm2DUB = 0;
cudaStatus = cudaMalloc(&dev_Mdm2DUB, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2DUB, &Mdm2DUB, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2DUB_global;
cudaStatus = cudaMalloc(&Mdm2DUB_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float DUB = 200.000000;
float* dev_DUB = 0;
cudaStatus = cudaMalloc(&dev_DUB, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_DUB, &DUB, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* DUB_global;
cudaStatus = cudaMalloc(&DUB_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Mdm2_p53_Ub = 0.000000;
float* dev_Mdm2_p53_Ub = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_p53_Ub, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_p53_Ub, &Mdm2_p53_Ub, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2_p53_Ub_global;
cudaStatus = cudaMalloc(&Mdm2_p53_Ub_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Mdm2_p53_Ub2 = 0.000000;
float* dev_Mdm2_p53_Ub2 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_p53_Ub2, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_p53_Ub2, &Mdm2_p53_Ub2, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2_p53_Ub2_global;
cudaStatus = cudaMalloc(&Mdm2_p53_Ub2_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Mdm2_p53_Ub3 = 0.000000;
float* dev_Mdm2_p53_Ub3 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_p53_Ub3, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_p53_Ub3, &Mdm2_p53_Ub3, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2_p53_Ub3_global;
cudaStatus = cudaMalloc(&Mdm2_p53_Ub3_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Mdm2_p53_Ub4 = 0.000000;
float* dev_Mdm2_p53_Ub4 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_p53_Ub4, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_p53_Ub4, &Mdm2_p53_Ub4, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2_p53_Ub4_global;
cudaStatus = cudaMalloc(&Mdm2_p53_Ub4_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Mdm2_P1_p53_Ub4 = 0.000000;
float* dev_Mdm2_P1_p53_Ub4 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_P1_p53_Ub4, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_P1_p53_Ub4, &Mdm2_P1_p53_Ub4, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2_P1_p53_Ub4_global;
cudaStatus = cudaMalloc(&Mdm2_P1_p53_Ub4_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Mdm2_Ub = 0.000000;
float* dev_Mdm2_Ub = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_Ub, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_Ub, &Mdm2_Ub, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2_Ub_global;
cudaStatus = cudaMalloc(&Mdm2_Ub_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Mdm2_Ub2 = 0.000000;
float* dev_Mdm2_Ub2 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_Ub2, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_Ub2, &Mdm2_Ub2, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2_Ub2_global;
cudaStatus = cudaMalloc(&Mdm2_Ub2_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Mdm2_Ub3 = 0.000000;
float* dev_Mdm2_Ub3 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_Ub3, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_Ub3, &Mdm2_Ub3, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2_Ub3_global;
cudaStatus = cudaMalloc(&Mdm2_Ub3_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Mdm2_Ub4 = 0.000000;
float* dev_Mdm2_Ub4 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_Ub4, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_Ub4, &Mdm2_Ub4, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2_Ub4_global;
cudaStatus = cudaMalloc(&Mdm2_Ub4_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Mdm2_P_Ub = 0.000000;
float* dev_Mdm2_P_Ub = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_P_Ub, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_P_Ub, &Mdm2_P_Ub, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2_P_Ub_global;
cudaStatus = cudaMalloc(&Mdm2_P_Ub_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Mdm2_P_Ub2 = 0.000000;
float* dev_Mdm2_P_Ub2 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_P_Ub2, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_P_Ub2, &Mdm2_P_Ub2, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2_P_Ub2_global;
cudaStatus = cudaMalloc(&Mdm2_P_Ub2_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Mdm2_P_Ub3 = 0.000000;
float* dev_Mdm2_P_Ub3 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_P_Ub3, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_P_Ub3, &Mdm2_P_Ub3, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2_P_Ub3_global;
cudaStatus = cudaMalloc(&Mdm2_P_Ub3_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Mdm2_P_Ub4 = 0.000000;
float* dev_Mdm2_P_Ub4 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_P_Ub4, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_P_Ub4, &Mdm2_P_Ub4, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2_P_Ub4_global;
cudaStatus = cudaMalloc(&Mdm2_P_Ub4_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float p53_Ub4_Proteasome = 0.000000;
float* dev_p53_Ub4_Proteasome = 0;
cudaStatus = cudaMalloc(&dev_p53_Ub4_Proteasome, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_p53_Ub4_Proteasome, &p53_Ub4_Proteasome, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* p53_Ub4_Proteasome_global;
cudaStatus = cudaMalloc(&p53_Ub4_Proteasome_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Mdm2_Ub4_Proteasome = 0.000000;
float* dev_Mdm2_Ub4_Proteasome = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_Ub4_Proteasome, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_Ub4_Proteasome, &Mdm2_Ub4_Proteasome, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2_Ub4_Proteasome_global;
cudaStatus = cudaMalloc(&Mdm2_Ub4_Proteasome_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Mdm2_P_Ub4_Proteasome = 0.000000;
float* dev_Mdm2_P_Ub4_Proteasome = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_P_Ub4_Proteasome, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_P_Ub4_Proteasome, &Mdm2_P_Ub4_Proteasome, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Mdm2_P_Ub4_Proteasome_global;
cudaStatus = cudaMalloc(&Mdm2_P_Ub4_Proteasome_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float GSK3b = 500.000000;
float* dev_GSK3b = 0;
cudaStatus = cudaMalloc(&dev_GSK3b, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_GSK3b, &GSK3b, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* GSK3b_global;
cudaStatus = cudaMalloc(&GSK3b_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float GSK3b_p53 = 0.000000;
float* dev_GSK3b_p53 = 0;
cudaStatus = cudaMalloc(&dev_GSK3b_p53, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_GSK3b_p53, &GSK3b_p53, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* GSK3b_p53_global;
cudaStatus = cudaMalloc(&GSK3b_p53_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float GSK3b_p53_P = 0.000000;
float* dev_GSK3b_p53_P = 0;
cudaStatus = cudaMalloc(&dev_GSK3b_p53_P, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_GSK3b_p53_P, &GSK3b_p53_P, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* GSK3b_p53_P_global;
cudaStatus = cudaMalloc(&GSK3b_p53_P_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Abeta = 0.000000;
float* dev_Abeta = 0;
cudaStatus = cudaMalloc(&dev_Abeta, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Abeta, &Abeta, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Abeta_global;
cudaStatus = cudaMalloc(&Abeta_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float AggAbeta_Proteasome = 0.000000;
float* dev_AggAbeta_Proteasome = 0;
cudaStatus = cudaMalloc(&dev_AggAbeta_Proteasome, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_AggAbeta_Proteasome, &AggAbeta_Proteasome, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* AggAbeta_Proteasome_global;
cudaStatus = cudaMalloc(&AggAbeta_Proteasome_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float AbetaPlaque = 0.000000;
float* dev_AbetaPlaque = 0;
cudaStatus = cudaMalloc(&dev_AbetaPlaque, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_AbetaPlaque, &AbetaPlaque, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* AbetaPlaque_global;
cudaStatus = cudaMalloc(&AbetaPlaque_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Tau = 0.000000;
float* dev_Tau = 0;
cudaStatus = cudaMalloc(&dev_Tau, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Tau, &Tau, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Tau_global;
cudaStatus = cudaMalloc(&Tau_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Tau_P1 = 0.000000;
float* dev_Tau_P1 = 0;
cudaStatus = cudaMalloc(&dev_Tau_P1, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Tau_P1, &Tau_P1, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Tau_P1_global;
cudaStatus = cudaMalloc(&Tau_P1_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Tau_P2 = 0.000000;
float* dev_Tau_P2 = 0;
cudaStatus = cudaMalloc(&dev_Tau_P2, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Tau_P2, &Tau_P2, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Tau_P2_global;
cudaStatus = cudaMalloc(&Tau_P2_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float MT_Tau = 100.000000;
float* dev_MT_Tau = 0;
cudaStatus = cudaMalloc(&dev_MT_Tau, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_MT_Tau, &MT_Tau, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* MT_Tau_global;
cudaStatus = cudaMalloc(&MT_Tau_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float AggTau = 0.000000;
float* dev_AggTau = 0;
cudaStatus = cudaMalloc(&dev_AggTau, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_AggTau, &AggTau, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* AggTau_global;
cudaStatus = cudaMalloc(&AggTau_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float AggTau_Proteasome = 0.000000;
float* dev_AggTau_Proteasome = 0;
cudaStatus = cudaMalloc(&dev_AggTau_Proteasome, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_AggTau_Proteasome, &AggTau_Proteasome, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* AggTau_Proteasome_global;
cudaStatus = cudaMalloc(&AggTau_Proteasome_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Proteasome_Tau = 0.000000;
float* dev_Proteasome_Tau = 0;
cudaStatus = cudaMalloc(&dev_Proteasome_Tau, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Proteasome_Tau, &Proteasome_Tau, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Proteasome_Tau_global;
cudaStatus = cudaMalloc(&Proteasome_Tau_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float PP1 = 50.000000;
float* dev_PP1 = 0;
cudaStatus = cudaMalloc(&dev_PP1, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_PP1, &PP1, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* PP1_global;
cudaStatus = cudaMalloc(&PP1_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float NFT = 0.000000;
float* dev_NFT = 0;
cudaStatus = cudaMalloc(&dev_NFT, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_NFT, &NFT, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* NFT_global;
cudaStatus = cudaMalloc(&NFT_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float ATP = 10000.000000;
float* dev_ATP = 0;
cudaStatus = cudaMalloc(&dev_ATP, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_ATP, &ATP, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* ATP_global;
cudaStatus = cudaMalloc(&ATP_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float ADP = 1000.000000;
float* dev_ADP = 0;
cudaStatus = cudaMalloc(&dev_ADP, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_ADP, &ADP, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* ADP_global;
cudaStatus = cudaMalloc(&ADP_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float AMP = 1000.000000;
float* dev_AMP = 0;
cudaStatus = cudaMalloc(&dev_AMP, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_AMP, &AMP, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* AMP_global;
cudaStatus = cudaMalloc(&AMP_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float AbetaDimer = 0.000000;
float* dev_AbetaDimer = 0;
cudaStatus = cudaMalloc(&dev_AbetaDimer, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_AbetaDimer, &AbetaDimer, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* AbetaDimer_global;
cudaStatus = cudaMalloc(&AbetaDimer_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float AbetaPlaque_GliaA = 0.000000;
float* dev_AbetaPlaque_GliaA = 0;
cudaStatus = cudaMalloc(&dev_AbetaPlaque_GliaA, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_AbetaPlaque_GliaA, &AbetaPlaque_GliaA, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* AbetaPlaque_GliaA_global;
cudaStatus = cudaMalloc(&AbetaPlaque_GliaA_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float GliaI = 100.000000;
float* dev_GliaI = 0;
cudaStatus = cudaMalloc(&dev_GliaI, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_GliaI, &GliaI, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* GliaI_global;
cudaStatus = cudaMalloc(&GliaI_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float GliaM1 = 0.000000;
float* dev_GliaM1 = 0;
cudaStatus = cudaMalloc(&dev_GliaM1, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_GliaM1, &GliaM1, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* GliaM1_global;
cudaStatus = cudaMalloc(&GliaM1_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float GliaM2 = 0.000000;
float* dev_GliaM2 = 0;
cudaStatus = cudaMalloc(&dev_GliaM2, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_GliaM2, &GliaM2, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* GliaM2_global;
cudaStatus = cudaMalloc(&GliaM2_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float GliaA = 0.000000;
float* dev_GliaA = 0;
cudaStatus = cudaMalloc(&dev_GliaA, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_GliaA, &GliaA, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* GliaA_global;
cudaStatus = cudaMalloc(&GliaA_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float antiAb = 0.000000;
float* dev_antiAb = 0;
cudaStatus = cudaMalloc(&dev_antiAb, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_antiAb, &antiAb, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* antiAb_global;
cudaStatus = cudaMalloc(&antiAb_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Abeta_antiAb = 0.000000;
float* dev_Abeta_antiAb = 0;
cudaStatus = cudaMalloc(&dev_Abeta_antiAb, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Abeta_antiAb, &Abeta_antiAb, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Abeta_antiAb_global;
cudaStatus = cudaMalloc(&Abeta_antiAb_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float AbetaDimer_antiAb = 0.000000;
float* dev_AbetaDimer_antiAb = 0;
cudaStatus = cudaMalloc(&dev_AbetaDimer_antiAb, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_AbetaDimer_antiAb, &AbetaDimer_antiAb, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* AbetaDimer_antiAb_global;
cudaStatus = cudaMalloc(&AbetaDimer_antiAb_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float degAbetaGlia = 0.000000;
float* dev_degAbetaGlia = 0;
cudaStatus = cudaMalloc(&dev_degAbetaGlia, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_degAbetaGlia, &degAbetaGlia, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* degAbetaGlia_global;
cudaStatus = cudaMalloc(&degAbetaGlia_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float disaggPlaque1 = 0.000000;
float* dev_disaggPlaque1 = 0;
cudaStatus = cudaMalloc(&dev_disaggPlaque1, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_disaggPlaque1, &disaggPlaque1, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* disaggPlaque1_global;
cudaStatus = cudaMalloc(&disaggPlaque1_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float disaggPlaque2 = 0.000000;
float* dev_disaggPlaque2 = 0;
cudaStatus = cudaMalloc(&dev_disaggPlaque2, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_disaggPlaque2, &disaggPlaque2, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* disaggPlaque2_global;
cudaStatus = cudaMalloc(&disaggPlaque2_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Source = 1.000000;
float* dev_Source = 0;
cudaStatus = cudaMalloc(&dev_Source, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Source, &Source, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Source_global;
cudaStatus = cudaMalloc(&Source_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
float Sink = 1.000000;
float* dev_Sink = 0;
cudaStatus = cudaMalloc(&dev_Sink, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Sink, &Sink, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float* Sink_global;
cudaStatus = cudaMalloc(&Sink_global, 32*sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
curandState *devStates;
CUDA_CALL(cudaMalloc((void **)&devStates, 32 * sizeof(curandState)));
initCurand<<<1, 32>>>(devStates, SEED);
for(int i = 0; i < 2; i++){
simulate<<<1, 32>>>(i, dev_output, devStates, 60.000000, 2000.000000, 1000, dev_Mdm2, Mdm2_global, dev_p53, p53_global, dev_Mdm2_p53, Mdm2_p53_global, dev_Mdm2_mRNA, Mdm2_mRNA_global, dev_p53_mRNA, p53_mRNA_global, dev_ATMA, ATMA_global, dev_ATMI, ATMI_global, dev_p53_P, p53_P_global, dev_Mdm2_P, Mdm2_P_global, dev_IR, IR_global, dev_ROS, ROS_global, dev_damDNA, damDNA_global, dev_E1, E1_global, dev_E2, E2_global, dev_E1_Ub, E1_Ub_global, dev_E2_Ub, E2_Ub_global, dev_Proteasome, Proteasome_global, dev_Ub, Ub_global, dev_p53DUB, p53DUB_global, dev_Mdm2DUB, Mdm2DUB_global, dev_DUB, DUB_global, dev_Mdm2_p53_Ub, Mdm2_p53_Ub_global, dev_Mdm2_p53_Ub2, Mdm2_p53_Ub2_global, dev_Mdm2_p53_Ub3, Mdm2_p53_Ub3_global, dev_Mdm2_p53_Ub4, Mdm2_p53_Ub4_global, dev_Mdm2_P1_p53_Ub4, Mdm2_P1_p53_Ub4_global, dev_Mdm2_Ub, Mdm2_Ub_global, dev_Mdm2_Ub2, Mdm2_Ub2_global, dev_Mdm2_Ub3, Mdm2_Ub3_global, dev_Mdm2_Ub4, Mdm2_Ub4_global, dev_Mdm2_P_Ub, Mdm2_P_Ub_global, dev_Mdm2_P_Ub2, Mdm2_P_Ub2_global, dev_Mdm2_P_Ub3, Mdm2_P_Ub3_global, dev_Mdm2_P_Ub4, Mdm2_P_Ub4_global, dev_p53_Ub4_Proteasome, p53_Ub4_Proteasome_global, dev_Mdm2_Ub4_Proteasome, Mdm2_Ub4_Proteasome_global, dev_Mdm2_P_Ub4_Proteasome, Mdm2_P_Ub4_Proteasome_global, dev_GSK3b, GSK3b_global, dev_GSK3b_p53, GSK3b_p53_global, dev_GSK3b_p53_P, GSK3b_p53_P_global, dev_Abeta, Abeta_global, dev_AggAbeta_Proteasome, AggAbeta_Proteasome_global, dev_AbetaPlaque, AbetaPlaque_global, dev_Tau, Tau_global, dev_Tau_P1, Tau_P1_global, dev_Tau_P2, Tau_P2_global, dev_MT_Tau, MT_Tau_global, dev_AggTau, AggTau_global, dev_AggTau_Proteasome, AggTau_Proteasome_global, dev_Proteasome_Tau, Proteasome_Tau_global, dev_PP1, PP1_global, dev_NFT, NFT_global, dev_ATP, ATP_global, dev_ADP, ADP_global, dev_AMP, AMP_global, dev_AbetaDimer, AbetaDimer_global, dev_AbetaPlaque_GliaA, AbetaPlaque_GliaA_global, dev_GliaI, GliaI_global, dev_GliaM1, GliaM1_global, dev_GliaM2, GliaM2_global, dev_GliaA, GliaA_global, dev_antiAb, antiAb_global, dev_Abeta_antiAb, Abeta_antiAb_global, dev_AbetaDimer_antiAb, AbetaDimer_antiAb_global, dev_degAbetaGlia, degAbetaGlia_global, dev_disaggPlaque1, disaggPlaque1_global, dev_disaggPlaque2, disaggPlaque2_global, dev_Source, Source_global, dev_Sink, Sink_global);

}
cudaStatus = cudaGetLastError(); if (cudaStatus != cudaSuccess) {fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));goto Error;}

cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);goto Error;}

cudaStatus = cudaMemcpy(output, dev_output, 34*69*sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2, dev_Mdm2, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&p53, dev_p53, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2_p53, dev_Mdm2_p53, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2_mRNA, dev_Mdm2_mRNA, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&p53_mRNA, dev_p53_mRNA, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&ATMA, dev_ATMA, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&ATMI, dev_ATMI, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&p53_P, dev_p53_P, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2_P, dev_Mdm2_P, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&IR, dev_IR, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&ROS, dev_ROS, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&damDNA, dev_damDNA, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&E1, dev_E1, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&E2, dev_E2, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&E1_Ub, dev_E1_Ub, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&E2_Ub, dev_E2_Ub, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Proteasome, dev_Proteasome, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Ub, dev_Ub, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&p53DUB, dev_p53DUB, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2DUB, dev_Mdm2DUB, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&DUB, dev_DUB, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2_p53_Ub, dev_Mdm2_p53_Ub, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2_p53_Ub2, dev_Mdm2_p53_Ub2, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2_p53_Ub3, dev_Mdm2_p53_Ub3, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2_p53_Ub4, dev_Mdm2_p53_Ub4, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2_P1_p53_Ub4, dev_Mdm2_P1_p53_Ub4, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2_Ub, dev_Mdm2_Ub, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2_Ub2, dev_Mdm2_Ub2, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2_Ub3, dev_Mdm2_Ub3, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2_Ub4, dev_Mdm2_Ub4, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2_P_Ub, dev_Mdm2_P_Ub, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2_P_Ub2, dev_Mdm2_P_Ub2, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2_P_Ub3, dev_Mdm2_P_Ub3, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2_P_Ub4, dev_Mdm2_P_Ub4, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&p53_Ub4_Proteasome, dev_p53_Ub4_Proteasome, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2_Ub4_Proteasome, dev_Mdm2_Ub4_Proteasome, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Mdm2_P_Ub4_Proteasome, dev_Mdm2_P_Ub4_Proteasome, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&GSK3b, dev_GSK3b, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&GSK3b_p53, dev_GSK3b_p53, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&GSK3b_p53_P, dev_GSK3b_p53_P, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Abeta, dev_Abeta, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&AggAbeta_Proteasome, dev_AggAbeta_Proteasome, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&AbetaPlaque, dev_AbetaPlaque, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Tau, dev_Tau, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Tau_P1, dev_Tau_P1, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Tau_P2, dev_Tau_P2, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&MT_Tau, dev_MT_Tau, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&AggTau, dev_AggTau, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&AggTau_Proteasome, dev_AggTau_Proteasome, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Proteasome_Tau, dev_Proteasome_Tau, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&PP1, dev_PP1, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&NFT, dev_NFT, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&ATP, dev_ATP, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&ADP, dev_ADP, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&AMP, dev_AMP, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&AbetaDimer, dev_AbetaDimer, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&AbetaPlaque_GliaA, dev_AbetaPlaque_GliaA, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&GliaI, dev_GliaI, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&GliaM1, dev_GliaM1, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&GliaM2, dev_GliaM2, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&GliaA, dev_GliaA, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&antiAb, dev_antiAb, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Abeta_antiAb, dev_Abeta_antiAb, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&AbetaDimer_antiAb, dev_AbetaDimer_antiAb, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&degAbetaGlia, dev_degAbetaGlia, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&disaggPlaque1, dev_disaggPlaque1, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&disaggPlaque2, dev_disaggPlaque2, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Source, dev_Source, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaMemcpy(&Sink, dev_Sink, sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
cudaStatus = cudaGetLastError(); if (cudaStatus != cudaSuccess) {fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));goto Error;}

cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);goto Error;}FILE* results = fopen("results.csv", "w");
if(results == NULL){
printf("Error acesssing results!");
exit(1);
}
fprintf(results, "time");
fprintf(results, ", Mdm2");
fprintf(results, ", p53");
fprintf(results, ", Mdm2_p53");
fprintf(results, ", Mdm2_mRNA");
fprintf(results, ", p53_mRNA");
fprintf(results, ", ATMA");
fprintf(results, ", ATMI");
fprintf(results, ", p53_P");
fprintf(results, ", Mdm2_P");
fprintf(results, ", IR");
fprintf(results, ", ROS");
fprintf(results, ", damDNA");
fprintf(results, ", E1");
fprintf(results, ", E2");
fprintf(results, ", E1_Ub");
fprintf(results, ", E2_Ub");
fprintf(results, ", Proteasome");
fprintf(results, ", Ub");
fprintf(results, ", p53DUB");
fprintf(results, ", Mdm2DUB");
fprintf(results, ", DUB");
fprintf(results, ", Mdm2_p53_Ub");
fprintf(results, ", Mdm2_p53_Ub2");
fprintf(results, ", Mdm2_p53_Ub3");
fprintf(results, ", Mdm2_p53_Ub4");
fprintf(results, ", Mdm2_P1_p53_Ub4");
fprintf(results, ", Mdm2_Ub");
fprintf(results, ", Mdm2_Ub2");
fprintf(results, ", Mdm2_Ub3");
fprintf(results, ", Mdm2_Ub4");
fprintf(results, ", Mdm2_P_Ub");
fprintf(results, ", Mdm2_P_Ub2");
fprintf(results, ", Mdm2_P_Ub3");
fprintf(results, ", Mdm2_P_Ub4");
fprintf(results, ", p53_Ub4_Proteasome");
fprintf(results, ", Mdm2_Ub4_Proteasome");
fprintf(results, ", Mdm2_P_Ub4_Proteasome");
fprintf(results, ", GSK3b");
fprintf(results, ", GSK3b_p53");
fprintf(results, ", GSK3b_p53_P");
fprintf(results, ", Abeta");
fprintf(results, ", AggAbeta_Proteasome");
fprintf(results, ", AbetaPlaque");
fprintf(results, ", Tau");
fprintf(results, ", Tau_P1");
fprintf(results, ", Tau_P2");
fprintf(results, ", MT_Tau");
fprintf(results, ", AggTau");
fprintf(results, ", AggTau_Proteasome");
fprintf(results, ", Proteasome_Tau");
fprintf(results, ", PP1");
fprintf(results, ", NFT");
fprintf(results, ", ATP");
fprintf(results, ", ADP");
fprintf(results, ", AMP");
fprintf(results, ", AbetaDimer");
fprintf(results, ", AbetaPlaque_GliaA");
fprintf(results, ", GliaI");
fprintf(results, ", GliaM1");
fprintf(results, ", GliaM2");
fprintf(results, ", GliaA");
fprintf(results, ", antiAb");
fprintf(results, ", Abeta_antiAb");
fprintf(results, ", AbetaDimer_antiAb");
fprintf(results, ", degAbetaGlia");
fprintf(results, ", disaggPlaque1");
fprintf(results, ", disaggPlaque2");
fprintf(results, ", Source");
fprintf(results, ", Sink");
fprintf(results, "\n");
for(int i = 0; i < 34; i++){
fprintf(results, "%lf", 60.000000*i);
for(int j = 0; j < 69; j++){
fprintf(results, ", %lf", output[69*i+j]/32);
}
fprintf(results, "\n");
}
fprintf(results, "\n");
Error:
cudaFree(dev_output);
cudaFree(dev_Mdm2);
cudaFree(Mdm2_global);
cudaFree(dev_p53);
cudaFree(p53_global);
cudaFree(dev_Mdm2_p53);
cudaFree(Mdm2_p53_global);
cudaFree(dev_Mdm2_mRNA);
cudaFree(Mdm2_mRNA_global);
cudaFree(dev_p53_mRNA);
cudaFree(p53_mRNA_global);
cudaFree(dev_ATMA);
cudaFree(ATMA_global);
cudaFree(dev_ATMI);
cudaFree(ATMI_global);
cudaFree(dev_p53_P);
cudaFree(p53_P_global);
cudaFree(dev_Mdm2_P);
cudaFree(Mdm2_P_global);
cudaFree(dev_IR);
cudaFree(IR_global);
cudaFree(dev_ROS);
cudaFree(ROS_global);
cudaFree(dev_damDNA);
cudaFree(damDNA_global);
cudaFree(dev_E1);
cudaFree(E1_global);
cudaFree(dev_E2);
cudaFree(E2_global);
cudaFree(dev_E1_Ub);
cudaFree(E1_Ub_global);
cudaFree(dev_E2_Ub);
cudaFree(E2_Ub_global);
cudaFree(dev_Proteasome);
cudaFree(Proteasome_global);
cudaFree(dev_Ub);
cudaFree(Ub_global);
cudaFree(dev_p53DUB);
cudaFree(p53DUB_global);
cudaFree(dev_Mdm2DUB);
cudaFree(Mdm2DUB_global);
cudaFree(dev_DUB);
cudaFree(DUB_global);
cudaFree(dev_Mdm2_p53_Ub);
cudaFree(Mdm2_p53_Ub_global);
cudaFree(dev_Mdm2_p53_Ub2);
cudaFree(Mdm2_p53_Ub2_global);
cudaFree(dev_Mdm2_p53_Ub3);
cudaFree(Mdm2_p53_Ub3_global);
cudaFree(dev_Mdm2_p53_Ub4);
cudaFree(Mdm2_p53_Ub4_global);
cudaFree(dev_Mdm2_P1_p53_Ub4);
cudaFree(Mdm2_P1_p53_Ub4_global);
cudaFree(dev_Mdm2_Ub);
cudaFree(Mdm2_Ub_global);
cudaFree(dev_Mdm2_Ub2);
cudaFree(Mdm2_Ub2_global);
cudaFree(dev_Mdm2_Ub3);
cudaFree(Mdm2_Ub3_global);
cudaFree(dev_Mdm2_Ub4);
cudaFree(Mdm2_Ub4_global);
cudaFree(dev_Mdm2_P_Ub);
cudaFree(Mdm2_P_Ub_global);
cudaFree(dev_Mdm2_P_Ub2);
cudaFree(Mdm2_P_Ub2_global);
cudaFree(dev_Mdm2_P_Ub3);
cudaFree(Mdm2_P_Ub3_global);
cudaFree(dev_Mdm2_P_Ub4);
cudaFree(Mdm2_P_Ub4_global);
cudaFree(dev_p53_Ub4_Proteasome);
cudaFree(p53_Ub4_Proteasome_global);
cudaFree(dev_Mdm2_Ub4_Proteasome);
cudaFree(Mdm2_Ub4_Proteasome_global);
cudaFree(dev_Mdm2_P_Ub4_Proteasome);
cudaFree(Mdm2_P_Ub4_Proteasome_global);
cudaFree(dev_GSK3b);
cudaFree(GSK3b_global);
cudaFree(dev_GSK3b_p53);
cudaFree(GSK3b_p53_global);
cudaFree(dev_GSK3b_p53_P);
cudaFree(GSK3b_p53_P_global);
cudaFree(dev_Abeta);
cudaFree(Abeta_global);
cudaFree(dev_AggAbeta_Proteasome);
cudaFree(AggAbeta_Proteasome_global);
cudaFree(dev_AbetaPlaque);
cudaFree(AbetaPlaque_global);
cudaFree(dev_Tau);
cudaFree(Tau_global);
cudaFree(dev_Tau_P1);
cudaFree(Tau_P1_global);
cudaFree(dev_Tau_P2);
cudaFree(Tau_P2_global);
cudaFree(dev_MT_Tau);
cudaFree(MT_Tau_global);
cudaFree(dev_AggTau);
cudaFree(AggTau_global);
cudaFree(dev_AggTau_Proteasome);
cudaFree(AggTau_Proteasome_global);
cudaFree(dev_Proteasome_Tau);
cudaFree(Proteasome_Tau_global);
cudaFree(dev_PP1);
cudaFree(PP1_global);
cudaFree(dev_NFT);
cudaFree(NFT_global);
cudaFree(dev_ATP);
cudaFree(ATP_global);
cudaFree(dev_ADP);
cudaFree(ADP_global);
cudaFree(dev_AMP);
cudaFree(AMP_global);
cudaFree(dev_AbetaDimer);
cudaFree(AbetaDimer_global);
cudaFree(dev_AbetaPlaque_GliaA);
cudaFree(AbetaPlaque_GliaA_global);
cudaFree(dev_GliaI);
cudaFree(GliaI_global);
cudaFree(dev_GliaM1);
cudaFree(GliaM1_global);
cudaFree(dev_GliaM2);
cudaFree(GliaM2_global);
cudaFree(dev_GliaA);
cudaFree(GliaA_global);
cudaFree(dev_antiAb);
cudaFree(antiAb_global);
cudaFree(dev_Abeta_antiAb);
cudaFree(Abeta_antiAb_global);
cudaFree(dev_AbetaDimer_antiAb);
cudaFree(AbetaDimer_antiAb_global);
cudaFree(dev_degAbetaGlia);
cudaFree(degAbetaGlia_global);
cudaFree(dev_disaggPlaque1);
cudaFree(disaggPlaque1_global);
cudaFree(dev_disaggPlaque2);
cudaFree(disaggPlaque2_global);
cudaFree(dev_Source);
cudaFree(Source_global);
cudaFree(dev_Sink);
cudaFree(Sink_global);

    return 0;
}