#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define pow powf
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
#define simulateStepReaction(i) \
switch (i) { \
case 0: \
reactionRate = ksynp53mRNA * Source; \
atomicAdd(p53_mRNA_aux, step * 1.000000 * reactionRate); \
break; \
case 1: \
reactionRate = kdegp53mRNA * p53_mRNA; \
atomicAdd(p53_mRNA_aux, - step * 1.000000 * reactionRate); \
break; \
case 2: \
reactionRate = ksynMdm2 * Mdm2_mRNA; \
atomicAdd(Mdm2_mRNA_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_mRNA_aux, step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_aux, step * 1.000000 * reactionRate); \
break; \
case 3: \
reactionRate = ksynMdm2mRNA * p53; \
atomicAdd(p53_aux, - step * 1.000000 * reactionRate); \
atomicAdd(p53_aux, step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_mRNA_aux, step * 1.000000 * reactionRate); \
break; \
case 4: \
reactionRate = ksynMdm2mRNA * p53_P; \
atomicAdd(p53_P_aux, - step * 1.000000 * reactionRate); \
atomicAdd(p53_P_aux, step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_mRNA_aux, step * 1.000000 * reactionRate); \
break; \
case 5: \
reactionRate = ksynMdm2mRNAGSK3bp53 * GSK3b_p53; \
atomicAdd(GSK3b_p53_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_p53_aux, step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_mRNA_aux, step * 1.000000 * reactionRate); \
break; \
case 6: \
reactionRate = ksynMdm2mRNAGSK3bp53 * GSK3b_p53_P; \
atomicAdd(GSK3b_p53_P_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_p53_P_aux, step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_mRNA_aux, step * 1.000000 * reactionRate); \
break; \
case 7: \
reactionRate = kdegMdm2mRNA * Mdm2_mRNA; \
atomicAdd(Mdm2_mRNA_aux, - step * 1.000000 * reactionRate); \
break; \
case 8: \
reactionRate = kbinMdm2p53 * p53 * Mdm2; \
atomicAdd(p53_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_p53_aux, step * 1.000000 * reactionRate); \
break; \
case 9: \
reactionRate = krelMdm2p53 * Mdm2_p53; \
atomicAdd(Mdm2_p53_aux, - step * 1.000000 * reactionRate); \
atomicAdd(p53_aux, step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_aux, step * 1.000000 * reactionRate); \
break; \
case 10: \
reactionRate = kbinGSK3bp53 * GSK3b * p53; \
atomicAdd(GSK3b_aux, - step * 1.000000 * reactionRate); \
atomicAdd(p53_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_p53_aux, step * 1.000000 * reactionRate); \
break; \
case 11: \
reactionRate = krelGSK3bp53 * GSK3b_p53; \
atomicAdd(GSK3b_p53_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_aux, step * 1.000000 * reactionRate); \
atomicAdd(p53_aux, step * 1.000000 * reactionRate); \
break; \
case 12: \
reactionRate = kbinGSK3bp53 * GSK3b * p53_P; \
atomicAdd(GSK3b_aux, - step * 1.000000 * reactionRate); \
atomicAdd(p53_P_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_p53_P_aux, step * 1.000000 * reactionRate); \
break; \
case 13: \
reactionRate = krelGSK3bp53 * GSK3b_p53_P; \
atomicAdd(GSK3b_p53_P_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_aux, step * 1.000000 * reactionRate); \
atomicAdd(p53_P_aux, step * 1.000000 * reactionRate); \
break; \
case 14: \
reactionRate = kbinE1Ub * E1 * Ub * ATP / (5000 + ATP); \
atomicAdd(E1_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(E1_Ub_aux, step * 1.000000 * reactionRate); \
break; \
case 15: \
reactionRate = kbinE2Ub * E2 * E1_Ub; \
atomicAdd(E2_aux, - step * 1.000000 * reactionRate); \
atomicAdd(E1_Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(E2_Ub_aux, step * 1.000000 * reactionRate); \
atomicAdd(E1_aux, step * 1.000000 * reactionRate); \
break; \
case 16: \
reactionRate = kMdm2Ub * Mdm2 * E2_Ub; \
atomicAdd(Mdm2_aux, - step * 1.000000 * reactionRate); \
atomicAdd(E2_Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_Ub_aux, step * 1.000000 * reactionRate); \
atomicAdd(E2_aux, step * 1.000000 * reactionRate); \
break; \
case 17: \
reactionRate = kMdm2PolyUb * Mdm2_Ub * E2_Ub; \
atomicAdd(Mdm2_Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(E2_Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_Ub2_aux, step * 1.000000 * reactionRate); \
atomicAdd(E2_aux, step * 1.000000 * reactionRate); \
break; \
case 18: \
reactionRate = kMdm2PolyUb * Mdm2_Ub2 * E2_Ub; \
atomicAdd(Mdm2_Ub2_aux, - step * 1.000000 * reactionRate); \
atomicAdd(E2_Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_Ub3_aux, step * 1.000000 * reactionRate); \
atomicAdd(E2_aux, step * 1.000000 * reactionRate); \
break; \
case 19: \
reactionRate = kMdm2PolyUb * Mdm2_Ub3 * E2_Ub; \
atomicAdd(Mdm2_Ub3_aux, - step * 1.000000 * reactionRate); \
atomicAdd(E2_Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_Ub4_aux, step * 1.000000 * reactionRate); \
atomicAdd(E2_aux, step * 1.000000 * reactionRate); \
break; \
case 20: \
reactionRate = kactDUBMdm2 * Mdm2_Ub4 * Mdm2DUB; \
atomicAdd(Mdm2_Ub4_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2DUB_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_Ub3_aux, step * 1.000000 * reactionRate); \
atomicAdd(Mdm2DUB_aux, step * 1.000000 * reactionRate); \
atomicAdd(Ub_aux, step * 1.000000 * reactionRate); \
break; \
case 21: \
reactionRate = kactDUBMdm2 * Mdm2_Ub3 * Mdm2DUB; \
atomicAdd(Mdm2_Ub3_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2DUB_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_Ub2_aux, step * 1.000000 * reactionRate); \
atomicAdd(Mdm2DUB_aux, step * 1.000000 * reactionRate); \
atomicAdd(Ub_aux, step * 1.000000 * reactionRate); \
break; \
case 22: \
reactionRate = kactDUBMdm2 * Mdm2_Ub2 * Mdm2DUB; \
atomicAdd(Mdm2_Ub2_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2DUB_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_Ub_aux, step * 1.000000 * reactionRate); \
atomicAdd(Mdm2DUB_aux, step * 1.000000 * reactionRate); \
atomicAdd(Ub_aux, step * 1.000000 * reactionRate); \
break; \
case 23: \
reactionRate = kactDUBMdm2 * Mdm2_Ub * Mdm2DUB; \
atomicAdd(Mdm2_Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2DUB_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_aux, step * 1.000000 * reactionRate); \
atomicAdd(Mdm2DUB_aux, step * 1.000000 * reactionRate); \
atomicAdd(Ub_aux, step * 1.000000 * reactionRate); \
break; \
case 24: \
reactionRate = kbinProt * Mdm2_Ub4 * Proteasome; \
atomicAdd(Mdm2_Ub4_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Proteasome_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_Ub4_Proteasome_aux, step * 1.000000 * reactionRate); \
break; \
case 25: \
reactionRate = kdegMdm2 * Mdm2_Ub4_Proteasome * kproteff; \
atomicAdd(Mdm2_Ub4_Proteasome_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Proteasome_aux, step * 1.000000 * reactionRate); \
atomicAdd(Ub_aux, step * 4.000000 * reactionRate); \
break; \
case 26: \
reactionRate = ksynp53 * p53_mRNA; \
atomicAdd(p53_mRNA_aux, - step * 1.000000 * reactionRate); \
atomicAdd(p53_aux, step * 1.000000 * reactionRate); \
atomicAdd(p53_mRNA_aux, step * 1.000000 * reactionRate); \
break; \
case 27: \
reactionRate = kp53Ub * E2_Ub * Mdm2_p53; \
atomicAdd(E2_Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_p53_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_p53_Ub_aux, step * 1.000000 * reactionRate); \
atomicAdd(E2_aux, step * 1.000000 * reactionRate); \
break; \
case 28: \
reactionRate = kp53PolyUb * Mdm2_p53_Ub * E2_Ub; \
atomicAdd(Mdm2_p53_Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(E2_Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_p53_Ub2_aux, step * 1.000000 * reactionRate); \
atomicAdd(E2_aux, step * 1.000000 * reactionRate); \
break; \
case 29: \
reactionRate = kp53PolyUb * Mdm2_p53_Ub2 * E2_Ub; \
atomicAdd(Mdm2_p53_Ub2_aux, - step * 1.000000 * reactionRate); \
atomicAdd(E2_Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_p53_Ub3_aux, step * 1.000000 * reactionRate); \
atomicAdd(E2_aux, step * 1.000000 * reactionRate); \
break; \
case 30: \
reactionRate = kp53PolyUb * Mdm2_p53_Ub3 * E2_Ub; \
atomicAdd(Mdm2_p53_Ub3_aux, - step * 1.000000 * reactionRate); \
atomicAdd(E2_Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_p53_Ub4_aux, step * 1.000000 * reactionRate); \
atomicAdd(E2_aux, step * 1.000000 * reactionRate); \
break; \
case 31: \
reactionRate = kactDUBp53 * Mdm2_p53_Ub4 * p53DUB; \
atomicAdd(Mdm2_p53_Ub4_aux, - step * 1.000000 * reactionRate); \
atomicAdd(p53DUB_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_p53_Ub3_aux, step * 1.000000 * reactionRate); \
atomicAdd(p53DUB_aux, step * 1.000000 * reactionRate); \
atomicAdd(Ub_aux, step * 1.000000 * reactionRate); \
break; \
case 32: \
reactionRate = kactDUBp53 * Mdm2_p53_Ub3 * p53DUB; \
atomicAdd(Mdm2_p53_Ub3_aux, - step * 1.000000 * reactionRate); \
atomicAdd(p53DUB_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_p53_Ub2_aux, step * 1.000000 * reactionRate); \
atomicAdd(p53DUB_aux, step * 1.000000 * reactionRate); \
atomicAdd(Ub_aux, step * 1.000000 * reactionRate); \
break; \
case 33: \
reactionRate = kactDUBp53 * Mdm2_p53_Ub2 * p53DUB; \
atomicAdd(Mdm2_p53_Ub2_aux, - step * 1.000000 * reactionRate); \
atomicAdd(p53DUB_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_p53_Ub_aux, step * 1.000000 * reactionRate); \
atomicAdd(p53DUB_aux, step * 1.000000 * reactionRate); \
atomicAdd(Ub_aux, step * 1.000000 * reactionRate); \
break; \
case 34: \
reactionRate = kactDUBp53 * Mdm2_p53_Ub * p53DUB; \
atomicAdd(Mdm2_p53_Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(p53DUB_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_p53_aux, step * 1.000000 * reactionRate); \
atomicAdd(p53DUB_aux, step * 1.000000 * reactionRate); \
atomicAdd(Ub_aux, step * 1.000000 * reactionRate); \
break; \
case 35: \
reactionRate = kphosMdm2GSK3b * Mdm2_p53_Ub4 * GSK3b; \
atomicAdd(Mdm2_p53_Ub4_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_P1_p53_Ub4_aux, step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_aux, step * 1.000000 * reactionRate); \
break; \
case 36: \
reactionRate = kphosMdm2GSK3bp53 * Mdm2_p53_Ub4 * GSK3b_p53; \
atomicAdd(Mdm2_p53_Ub4_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_p53_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_P1_p53_Ub4_aux, step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_p53_aux, step * 1.000000 * reactionRate); \
break; \
case 37: \
reactionRate = kphosMdm2GSK3bp53 * Mdm2_p53_Ub4 * GSK3b_p53_P; \
atomicAdd(Mdm2_p53_Ub4_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_p53_P_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_P1_p53_Ub4_aux, step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_p53_P_aux, step * 1.000000 * reactionRate); \
break; \
case 38: \
reactionRate = kbinProt * Mdm2_P1_p53_Ub4 * Proteasome; \
atomicAdd(Mdm2_P1_p53_Ub4_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Proteasome_aux, - step * 1.000000 * reactionRate); \
atomicAdd(p53_Ub4_Proteasome_aux, step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_aux, step * 1.000000 * reactionRate); \
break; \
case 39: \
reactionRate = kdegp53 * kproteff * p53_Ub4_Proteasome * ATP / (5000 + ATP); \
atomicAdd(p53_Ub4_Proteasome_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Ub_aux, step * 4.000000 * reactionRate); \
atomicAdd(Proteasome_aux, step * 1.000000 * reactionRate); \
break; \
case 40: \
reactionRate = kbinMTTau * Tau; \
atomicAdd(Tau_aux, - step * 1.000000 * reactionRate); \
atomicAdd(MT_Tau_aux, step * 1.000000 * reactionRate); \
break; \
case 41: \
reactionRate = krelMTTau * MT_Tau; \
atomicAdd(MT_Tau_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Tau_aux, step * 1.000000 * reactionRate); \
break; \
case 42: \
reactionRate = kphospTauGSK3bp53 * GSK3b_p53 * Tau; \
atomicAdd(GSK3b_p53_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Tau_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_p53_aux, step * 1.000000 * reactionRate); \
atomicAdd(Tau_P1_aux, step * 1.000000 * reactionRate); \
break; \
case 43: \
reactionRate = kphospTauGSK3bp53 * GSK3b_p53 * Tau_P1; \
atomicAdd(GSK3b_p53_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Tau_P1_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_p53_aux, step * 1.000000 * reactionRate); \
atomicAdd(Tau_P2_aux, step * 1.000000 * reactionRate); \
break; \
case 44: \
reactionRate = kphospTauGSK3bp53 * GSK3b_p53_P * Tau; \
atomicAdd(GSK3b_p53_P_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Tau_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_p53_P_aux, step * 1.000000 * reactionRate); \
atomicAdd(Tau_P1_aux, step * 1.000000 * reactionRate); \
break; \
case 45: \
reactionRate = kphospTauGSK3bp53 * GSK3b_p53_P * Tau_P1; \
atomicAdd(GSK3b_p53_P_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Tau_P1_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_p53_P_aux, step * 1.000000 * reactionRate); \
atomicAdd(Tau_P2_aux, step * 1.000000 * reactionRate); \
break; \
case 46: \
reactionRate = kphospTauGSK3b * GSK3b * Tau; \
atomicAdd(GSK3b_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Tau_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_aux, step * 1.000000 * reactionRate); \
atomicAdd(Tau_P1_aux, step * 1.000000 * reactionRate); \
break; \
case 47: \
reactionRate = kphospTauGSK3b * GSK3b * Tau_P1; \
atomicAdd(GSK3b_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Tau_P1_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_aux, step * 1.000000 * reactionRate); \
atomicAdd(Tau_P2_aux, step * 1.000000 * reactionRate); \
break; \
case 48: \
reactionRate = kdephospTau * Tau_P2 * PP1; \
atomicAdd(Tau_P2_aux, - step * 1.000000 * reactionRate); \
atomicAdd(PP1_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Tau_P1_aux, step * 1.000000 * reactionRate); \
atomicAdd(PP1_aux, step * 1.000000 * reactionRate); \
break; \
case 49: \
reactionRate = kdephospTau * Tau_P1 * PP1; \
atomicAdd(Tau_P1_aux, - step * 1.000000 * reactionRate); \
atomicAdd(PP1_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Tau_aux, step * 1.000000 * reactionRate); \
atomicAdd(PP1_aux, step * 1.000000 * reactionRate); \
break; \
case 50: \
reactionRate = kaggTauP1 * Tau_P1 * (Tau_P1 - 1) * 0.5; \
atomicAdd(Tau_P1_aux, - step * 2.000000 * reactionRate); \
atomicAdd(AggTau_aux, step * 2.000000 * reactionRate); \
break; \
case 51: \
reactionRate = kaggTauP1 * Tau_P1 * AggTau; \
atomicAdd(Tau_P1_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AggTau_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AggTau_aux, step * 2.000000 * reactionRate); \
break; \
case 52: \
reactionRate = kaggTauP2 * Tau_P2 * (Tau_P2 - 1) * 0.5; \
atomicAdd(Tau_P2_aux, - step * 2.000000 * reactionRate); \
atomicAdd(AggTau_aux, step * 2.000000 * reactionRate); \
break; \
case 53: \
reactionRate = kaggTauP2 * Tau_P2 * AggTau; \
atomicAdd(Tau_P2_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AggTau_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AggTau_aux, step * 2.000000 * reactionRate); \
break; \
case 54: \
reactionRate = kaggTau * Tau * (Tau - 1) * 0.5; \
atomicAdd(Tau_aux, - step * 2.000000 * reactionRate); \
atomicAdd(AggTau_aux, step * 2.000000 * reactionRate); \
break; \
case 55: \
reactionRate = kaggTau * Tau * AggTau; \
atomicAdd(Tau_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AggTau_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AggTau_aux, step * 2.000000 * reactionRate); \
break; \
case 56: \
reactionRate = ktangfor * AggTau * (AggTau - 1) * 0.5; \
atomicAdd(AggTau_aux, - step * 2.000000 * reactionRate); \
atomicAdd(NFT_aux, step * 2.000000 * reactionRate); \
break; \
case 57: \
reactionRate = ktangfor * AggTau * NFT; \
atomicAdd(AggTau_aux, - step * 1.000000 * reactionRate); \
atomicAdd(NFT_aux, - step * 1.000000 * reactionRate); \
atomicAdd(NFT_aux, step * 2.000000 * reactionRate); \
break; \
case 58: \
reactionRate = kinhibprot * AggTau * Proteasome; \
atomicAdd(AggTau_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Proteasome_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AggTau_Proteasome_aux, step * 1.000000 * reactionRate); \
break; \
case 59: \
reactionRate = kprodAbeta * Source; \
atomicAdd(Abeta_aux, step * 1.000000 * reactionRate); \
break; \
case 60: \
reactionRate = kprodAbeta2 * GSK3b_p53; \
atomicAdd(GSK3b_p53_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Abeta_aux, step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_p53_aux, step * 1.000000 * reactionRate); \
break; \
case 61: \
reactionRate = kprodAbeta2 * GSK3b_p53_P; \
atomicAdd(GSK3b_p53_P_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Abeta_aux, step * 1.000000 * reactionRate); \
atomicAdd(GSK3b_p53_P_aux, step * 1.000000 * reactionRate); \
break; \
case 62: \
reactionRate = kinhibprot * AbetaDimer * Proteasome; \
atomicAdd(AbetaDimer_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Proteasome_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AggAbeta_Proteasome_aux, step * 1.000000 * reactionRate); \
break; \
case 63: \
reactionRate = kdegAbeta * Abeta; \
atomicAdd(Abeta_aux, - step * 1.000000 * reactionRate); \
break; \
case 64: \
reactionRate = ksynp53mRNAAbeta * Abeta; \
atomicAdd(Abeta_aux, - step * 1.000000 * reactionRate); \
atomicAdd(p53_mRNA_aux, step * 1.000000 * reactionRate); \
atomicAdd(Abeta_aux, step * 1.000000 * reactionRate); \
break; \
case 65: \
reactionRate = kdam * IR; \
atomicAdd(IR_aux, - step * 1.000000 * reactionRate); \
atomicAdd(IR_aux, step * 1.000000 * reactionRate); \
atomicAdd(damDNA_aux, step * 1.000000 * reactionRate); \
break; \
case 66: \
reactionRate = krepair * damDNA; \
atomicAdd(damDNA_aux, - step * 1.000000 * reactionRate); \
break; \
case 67: \
reactionRate = kactATM * damDNA * ATMI; \
atomicAdd(damDNA_aux, - step * 1.000000 * reactionRate); \
atomicAdd(ATMI_aux, - step * 1.000000 * reactionRate); \
atomicAdd(damDNA_aux, step * 1.000000 * reactionRate); \
atomicAdd(ATMA_aux, step * 1.000000 * reactionRate); \
break; \
case 68: \
reactionRate = kphosp53 * p53 * ATMA; \
atomicAdd(p53_aux, - step * 1.000000 * reactionRate); \
atomicAdd(ATMA_aux, - step * 1.000000 * reactionRate); \
atomicAdd(p53_P_aux, step * 1.000000 * reactionRate); \
atomicAdd(ATMA_aux, step * 1.000000 * reactionRate); \
break; \
case 69: \
reactionRate = kdephosp53 * p53_P; \
atomicAdd(p53_P_aux, - step * 1.000000 * reactionRate); \
atomicAdd(p53_aux, step * 1.000000 * reactionRate); \
break; \
case 70: \
reactionRate = kphosMdm2 * Mdm2 * ATMA; \
atomicAdd(Mdm2_aux, - step * 1.000000 * reactionRate); \
atomicAdd(ATMA_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_P_aux, step * 1.000000 * reactionRate); \
atomicAdd(ATMA_aux, step * 1.000000 * reactionRate); \
break; \
case 71: \
reactionRate = kdephosMdm2 * Mdm2_P; \
atomicAdd(Mdm2_P_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_aux, step * 1.000000 * reactionRate); \
break; \
case 72: \
reactionRate = kMdm2PUb * Mdm2_P * E2_Ub; \
atomicAdd(Mdm2_P_aux, - step * 1.000000 * reactionRate); \
atomicAdd(E2_Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_P_Ub_aux, step * 1.000000 * reactionRate); \
atomicAdd(E2_aux, step * 1.000000 * reactionRate); \
break; \
case 73: \
reactionRate = kMdm2PolyUb * Mdm2_P_Ub * E2_Ub; \
atomicAdd(Mdm2_P_Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(E2_Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_P_Ub2_aux, step * 1.000000 * reactionRate); \
atomicAdd(E2_aux, step * 1.000000 * reactionRate); \
break; \
case 74: \
reactionRate = kMdm2PolyUb * Mdm2_P_Ub2 * E2_Ub; \
atomicAdd(Mdm2_P_Ub2_aux, - step * 1.000000 * reactionRate); \
atomicAdd(E2_Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_P_Ub3_aux, step * 1.000000 * reactionRate); \
atomicAdd(E2_aux, step * 1.000000 * reactionRate); \
break; \
case 75: \
reactionRate = kMdm2PolyUb * Mdm2_P_Ub3 * E2_Ub; \
atomicAdd(Mdm2_P_Ub3_aux, - step * 1.000000 * reactionRate); \
atomicAdd(E2_Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_P_Ub4_aux, step * 1.000000 * reactionRate); \
atomicAdd(E2_aux, step * 1.000000 * reactionRate); \
break; \
case 76: \
reactionRate = kactDUBMdm2 * Mdm2_P_Ub4 * Mdm2DUB; \
atomicAdd(Mdm2_P_Ub4_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2DUB_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_P_Ub3_aux, step * 1.000000 * reactionRate); \
atomicAdd(Mdm2DUB_aux, step * 1.000000 * reactionRate); \
atomicAdd(Ub_aux, step * 1.000000 * reactionRate); \
break; \
case 77: \
reactionRate = kactDUBMdm2 * Mdm2_P_Ub3 * Mdm2DUB; \
atomicAdd(Mdm2_P_Ub3_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2DUB_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_P_Ub2_aux, step * 1.000000 * reactionRate); \
atomicAdd(Mdm2DUB_aux, step * 1.000000 * reactionRate); \
atomicAdd(Ub_aux, step * 1.000000 * reactionRate); \
break; \
case 78: \
reactionRate = kactDUBMdm2 * Mdm2_P_Ub2 * Mdm2DUB; \
atomicAdd(Mdm2_P_Ub2_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2DUB_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_P_Ub_aux, step * 1.000000 * reactionRate); \
atomicAdd(Mdm2DUB_aux, step * 1.000000 * reactionRate); \
atomicAdd(Ub_aux, step * 1.000000 * reactionRate); \
break; \
case 79: \
reactionRate = kactDUBMdm2 * Mdm2_P_Ub * Mdm2DUB; \
atomicAdd(Mdm2_P_Ub_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2DUB_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_P_aux, step * 1.000000 * reactionRate); \
atomicAdd(Mdm2DUB_aux, step * 1.000000 * reactionRate); \
atomicAdd(Ub_aux, step * 1.000000 * reactionRate); \
break; \
case 80: \
reactionRate = kbinProt * Mdm2_P_Ub4 * Proteasome; \
atomicAdd(Mdm2_P_Ub4_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Proteasome_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Mdm2_P_Ub4_Proteasome_aux, step * 1.000000 * reactionRate); \
break; \
case 81: \
reactionRate = kdegMdm2 * Mdm2_P_Ub4_Proteasome * kproteff; \
atomicAdd(Mdm2_P_Ub4_Proteasome_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Proteasome_aux, step * 1.000000 * reactionRate); \
atomicAdd(Ub_aux, step * 4.000000 * reactionRate); \
break; \
case 82: \
reactionRate = kinactATM * ATMA; \
atomicAdd(ATMA_aux, - step * 1.000000 * reactionRate); \
atomicAdd(ATMI_aux, step * 1.000000 * reactionRate); \
break; \
case 83: \
reactionRate = kgenROSAbeta * Abeta; \
atomicAdd(Abeta_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Abeta_aux, step * 1.000000 * reactionRate); \
atomicAdd(ROS_aux, step * 1.000000 * reactionRate); \
break; \
case 84: \
reactionRate = kgenROSPlaque * AbetaPlaque; \
atomicAdd(AbetaPlaque_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AbetaPlaque_aux, step * 1.000000 * reactionRate); \
atomicAdd(ROS_aux, step * 1.000000 * reactionRate); \
break; \
case 85: \
reactionRate = kgenROSAbeta * AggAbeta_Proteasome; \
atomicAdd(AggAbeta_Proteasome_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AggAbeta_Proteasome_aux, step * 1.000000 * reactionRate); \
atomicAdd(ROS_aux, step * 1.000000 * reactionRate); \
break; \
case 86: \
reactionRate = kdamROS * ROS; \
atomicAdd(ROS_aux, - step * 1.000000 * reactionRate); \
atomicAdd(ROS_aux, step * 1.000000 * reactionRate); \
atomicAdd(damDNA_aux, step * 1.000000 * reactionRate); \
break; \
case 87: \
reactionRate = ksynTau * Source; \
atomicAdd(Tau_aux, step * 1.000000 * reactionRate); \
break; \
case 88: \
reactionRate = kbinTauProt * Tau * Proteasome; \
atomicAdd(Tau_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Proteasome_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Proteasome_Tau_aux, step * 1.000000 * reactionRate); \
break; \
case 89: \
reactionRate = kdegTau20SProt * Proteasome_Tau; \
atomicAdd(Proteasome_Tau_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Proteasome_aux, step * 1.000000 * reactionRate); \
break; \
case 90: \
reactionRate = kaggAbeta * Abeta * (Abeta - 1) * 0.5; \
atomicAdd(Abeta_aux, - step * 2.000000 * reactionRate); \
atomicAdd(AbetaDimer_aux, step * 1.000000 * reactionRate); \
break; \
case 91: \
reactionRate = kpf * AbetaDimer * (AbetaDimer - 1) * 0.5; \
atomicAdd(AbetaDimer_aux, - step * 2.000000 * reactionRate); \
atomicAdd(AbetaPlaque_aux, step * 1.000000 * reactionRate); \
break; \
case 92: \
reactionRate = kpg * AbetaDimer * pow(AbetaPlaque, 2) / (pow(kpghalf, 2) + pow(AbetaPlaque, 2)); \
atomicAdd(AbetaDimer_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AbetaPlaque_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AbetaPlaque_aux, step * 2.000000 * reactionRate); \
break; \
case 93: \
reactionRate = kdisaggAbeta * AbetaDimer; \
atomicAdd(AbetaDimer_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Abeta_aux, step * 2.000000 * reactionRate); \
break; \
case 94: \
reactionRate = kdisaggAbeta1 * AbetaPlaque; \
atomicAdd(AbetaPlaque_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AbetaDimer_aux, step * 1.000000 * reactionRate); \
atomicAdd(disaggPlaque1_aux, step * 1.000000 * reactionRate); \
break; \
case 95: \
reactionRate = kdisaggAbeta2 * antiAb * AbetaPlaque; \
atomicAdd(AbetaPlaque_aux, - step * 1.000000 * reactionRate); \
atomicAdd(antiAb_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AbetaDimer_aux, step * 1.000000 * reactionRate); \
atomicAdd(antiAb_aux, step * 1.000000 * reactionRate); \
atomicAdd(disaggPlaque2_aux, step * 1.000000 * reactionRate); \
break; \
case 96: \
reactionRate = kbinAbantiAb * Abeta * antiAb; \
atomicAdd(Abeta_aux, - step * 1.000000 * reactionRate); \
atomicAdd(antiAb_aux, - step * 1.000000 * reactionRate); \
atomicAdd(Abeta_antiAb_aux, step * 1.000000 * reactionRate); \
break; \
case 97: \
reactionRate = kbinAbantiAb * AbetaDimer * antiAb; \
atomicAdd(AbetaDimer_aux, - step * 1.000000 * reactionRate); \
atomicAdd(antiAb_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AbetaDimer_antiAb_aux, step * 1.000000 * reactionRate); \
break; \
case 98: \
reactionRate = 10 * kdegAbeta * Abeta_antiAb; \
atomicAdd(Abeta_antiAb_aux, - step * 1.000000 * reactionRate); \
atomicAdd(antiAb_aux, step * 1.000000 * reactionRate); \
break; \
case 99: \
reactionRate = 10 * kdegAbeta * AbetaDimer_antiAb; \
atomicAdd(AbetaDimer_antiAb_aux, - step * 1.000000 * reactionRate); \
atomicAdd(antiAb_aux, step * 1.000000 * reactionRate); \
break; \
case 100: \
reactionRate = kactglia1 * GliaI * AbetaPlaque; \
atomicAdd(GliaI_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AbetaPlaque_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GliaM1_aux, step * 1.000000 * reactionRate); \
atomicAdd(AbetaPlaque_aux, step * 1.000000 * reactionRate); \
break; \
case 101: \
reactionRate = kactglia1 * GliaM1 * AbetaPlaque; \
atomicAdd(GliaM1_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AbetaPlaque_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GliaM2_aux, step * 1.000000 * reactionRate); \
atomicAdd(AbetaPlaque_aux, step * 1.000000 * reactionRate); \
break; \
case 102: \
reactionRate = kactglia2 * GliaM2 * antiAb; \
atomicAdd(GliaM2_aux, - step * 1.000000 * reactionRate); \
atomicAdd(antiAb_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GliaA_aux, step * 1.000000 * reactionRate); \
atomicAdd(antiAb_aux, step * 1.000000 * reactionRate); \
break; \
case 103: \
reactionRate = kinactglia1 * GliaA; \
atomicAdd(GliaA_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GliaM2_aux, step * 1.000000 * reactionRate); \
break; \
case 104: \
reactionRate = kinactglia2 * GliaM2; \
atomicAdd(GliaM2_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GliaM1_aux, step * 1.000000 * reactionRate); \
break; \
case 105: \
reactionRate = kinactglia2 * GliaM1; \
atomicAdd(GliaM1_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GliaI_aux, step * 1.000000 * reactionRate); \
break; \
case 106: \
reactionRate = kbinAbetaGlia * AbetaPlaque * GliaA; \
atomicAdd(AbetaPlaque_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GliaA_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AbetaPlaque_GliaA_aux, step * 1.000000 * reactionRate); \
break; \
case 107: \
reactionRate = krelAbetaGlia * AbetaPlaque_GliaA; \
atomicAdd(AbetaPlaque_GliaA_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AbetaPlaque_aux, step * 1.000000 * reactionRate); \
atomicAdd(GliaA_aux, step * 1.000000 * reactionRate); \
break; \
case 108: \
reactionRate = kdegAbetaGlia * AbetaPlaque_GliaA; \
atomicAdd(AbetaPlaque_GliaA_aux, - step * 1.000000 * reactionRate); \
atomicAdd(GliaA_aux, step * 1.000000 * reactionRate); \
atomicAdd(degAbetaGlia_aux, step * 1.000000 * reactionRate); \
break; \
case 109: \
reactionRate = kgenROSGlia * AbetaPlaque_GliaA; \
atomicAdd(AbetaPlaque_GliaA_aux, - step * 1.000000 * reactionRate); \
atomicAdd(AbetaPlaque_GliaA_aux, step * 1.000000 * reactionRate); \
atomicAdd(ROS_aux, step * 1.000000 * reactionRate); \
break; \
case 110: \
reactionRate = kdegAntiAb * antiAb; \
atomicAdd(antiAb_aux, - step * 1.000000 * reactionRate); \
break; \
case 111: \
reactionRate = kremROS * ROS; \
atomicAdd(ROS_aux, - step * 1.000000 * reactionRate); \
break; \
} \


__global__
void simulate (float step, int numSimulations, float Mdm2, float* Mdm2_aux, float p53, float* p53_aux, float Mdm2_p53, float* Mdm2_p53_aux, float Mdm2_mRNA, float* Mdm2_mRNA_aux, float p53_mRNA, float* p53_mRNA_aux, float ATMA, float* ATMA_aux, float ATMI, float* ATMI_aux, float p53_P, float* p53_P_aux, float Mdm2_P, float* Mdm2_P_aux, float IR, float* IR_aux, float ROS, float* ROS_aux, float damDNA, float* damDNA_aux, float E1, float* E1_aux, float E2, float* E2_aux, float E1_Ub, float* E1_Ub_aux, float E2_Ub, float* E2_Ub_aux, float Proteasome, float* Proteasome_aux, float Ub, float* Ub_aux, float p53DUB, float* p53DUB_aux, float Mdm2DUB, float* Mdm2DUB_aux, float DUB, float* DUB_aux, float Mdm2_p53_Ub, float* Mdm2_p53_Ub_aux, float Mdm2_p53_Ub2, float* Mdm2_p53_Ub2_aux, float Mdm2_p53_Ub3, float* Mdm2_p53_Ub3_aux, float Mdm2_p53_Ub4, float* Mdm2_p53_Ub4_aux, float Mdm2_P1_p53_Ub4, float* Mdm2_P1_p53_Ub4_aux, float Mdm2_Ub, float* Mdm2_Ub_aux, float Mdm2_Ub2, float* Mdm2_Ub2_aux, float Mdm2_Ub3, float* Mdm2_Ub3_aux, float Mdm2_Ub4, float* Mdm2_Ub4_aux, float Mdm2_P_Ub, float* Mdm2_P_Ub_aux, float Mdm2_P_Ub2, float* Mdm2_P_Ub2_aux, float Mdm2_P_Ub3, float* Mdm2_P_Ub3_aux, float Mdm2_P_Ub4, float* Mdm2_P_Ub4_aux, float p53_Ub4_Proteasome, float* p53_Ub4_Proteasome_aux, float Mdm2_Ub4_Proteasome, float* Mdm2_Ub4_Proteasome_aux, float Mdm2_P_Ub4_Proteasome, float* Mdm2_P_Ub4_Proteasome_aux, float GSK3b, float* GSK3b_aux, float GSK3b_p53, float* GSK3b_p53_aux, float GSK3b_p53_P, float* GSK3b_p53_P_aux, float Abeta, float* Abeta_aux, float AggAbeta_Proteasome, float* AggAbeta_Proteasome_aux, float AbetaPlaque, float* AbetaPlaque_aux, float Tau, float* Tau_aux, float Tau_P1, float* Tau_P1_aux, float Tau_P2, float* Tau_P2_aux, float MT_Tau, float* MT_Tau_aux, float AggTau, float* AggTau_aux, float AggTau_Proteasome, float* AggTau_Proteasome_aux, float Proteasome_Tau, float* Proteasome_Tau_aux, float PP1, float* PP1_aux, float NFT, float* NFT_aux, float ATP, float* ATP_aux, float ADP, float* ADP_aux, float AMP, float* AMP_aux, float AbetaDimer, float* AbetaDimer_aux, float AbetaPlaque_GliaA, float* AbetaPlaque_GliaA_aux, float GliaI, float* GliaI_aux, float GliaM1, float* GliaM1_aux, float GliaM2, float* GliaM2_aux, float GliaA, float* GliaA_aux, float antiAb, float* antiAb_aux, float Abeta_antiAb, float* Abeta_antiAb_aux, float AbetaDimer_antiAb, float* AbetaDimer_antiAb_aux, float degAbetaGlia, float* degAbetaGlia_aux, float disaggPlaque1, float* disaggPlaque1_aux, float disaggPlaque2, float* disaggPlaque2_aux, float Source, float* Source_aux, float Sink, float* Sink_aux) {
float reactionRate;
for(int i = 0; i < numSimulations; i++){
Mdm2 = *Mdm2_aux;
p53 = *p53_aux;
Mdm2_p53 = *Mdm2_p53_aux;
Mdm2_mRNA = *Mdm2_mRNA_aux;
p53_mRNA = *p53_mRNA_aux;
ATMA = *ATMA_aux;
ATMI = *ATMI_aux;
p53_P = *p53_P_aux;
Mdm2_P = *Mdm2_P_aux;
IR = *IR_aux;
ROS = *ROS_aux;
damDNA = *damDNA_aux;
E1 = *E1_aux;
E2 = *E2_aux;
E1_Ub = *E1_Ub_aux;
E2_Ub = *E2_Ub_aux;
Proteasome = *Proteasome_aux;
Ub = *Ub_aux;
p53DUB = *p53DUB_aux;
Mdm2DUB = *Mdm2DUB_aux;
DUB = *DUB_aux;
Mdm2_p53_Ub = *Mdm2_p53_Ub_aux;
Mdm2_p53_Ub2 = *Mdm2_p53_Ub2_aux;
Mdm2_p53_Ub3 = *Mdm2_p53_Ub3_aux;
Mdm2_p53_Ub4 = *Mdm2_p53_Ub4_aux;
Mdm2_P1_p53_Ub4 = *Mdm2_P1_p53_Ub4_aux;
Mdm2_Ub = *Mdm2_Ub_aux;
Mdm2_Ub2 = *Mdm2_Ub2_aux;
Mdm2_Ub3 = *Mdm2_Ub3_aux;
Mdm2_Ub4 = *Mdm2_Ub4_aux;
Mdm2_P_Ub = *Mdm2_P_Ub_aux;
Mdm2_P_Ub2 = *Mdm2_P_Ub2_aux;
Mdm2_P_Ub3 = *Mdm2_P_Ub3_aux;
Mdm2_P_Ub4 = *Mdm2_P_Ub4_aux;
p53_Ub4_Proteasome = *p53_Ub4_Proteasome_aux;
Mdm2_Ub4_Proteasome = *Mdm2_Ub4_Proteasome_aux;
Mdm2_P_Ub4_Proteasome = *Mdm2_P_Ub4_Proteasome_aux;
GSK3b = *GSK3b_aux;
GSK3b_p53 = *GSK3b_p53_aux;
GSK3b_p53_P = *GSK3b_p53_P_aux;
Abeta = *Abeta_aux;
AggAbeta_Proteasome = *AggAbeta_Proteasome_aux;
AbetaPlaque = *AbetaPlaque_aux;
Tau = *Tau_aux;
Tau_P1 = *Tau_P1_aux;
Tau_P2 = *Tau_P2_aux;
MT_Tau = *MT_Tau_aux;
AggTau = *AggTau_aux;
AggTau_Proteasome = *AggTau_Proteasome_aux;
Proteasome_Tau = *Proteasome_Tau_aux;
PP1 = *PP1_aux;
NFT = *NFT_aux;
ATP = *ATP_aux;
ADP = *ADP_aux;
AMP = *AMP_aux;
AbetaDimer = *AbetaDimer_aux;
AbetaPlaque_GliaA = *AbetaPlaque_GliaA_aux;
GliaI = *GliaI_aux;
GliaM1 = *GliaM1_aux;
GliaM2 = *GliaM2_aux;
GliaA = *GliaA_aux;
antiAb = *antiAb_aux;
Abeta_antiAb = *Abeta_antiAb_aux;
AbetaDimer_antiAb = *AbetaDimer_antiAb_aux;
degAbetaGlia = *degAbetaGlia_aux;
disaggPlaque1 = *disaggPlaque1_aux;
disaggPlaque2 = *disaggPlaque2_aux;
Source = *Source_aux;
Sink = *Sink_aux;
simulateStepReaction(threadIdx.x);
__syncthreads();
}
}

int main()
{
cudaError_t cudaStatus;
float Mdm2 = 5.000000;
float* dev_Mdm2 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2, &Mdm2, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float p53 = 5.000000;
float* dev_p53 = 0;
cudaStatus = cudaMalloc(&dev_p53, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_p53, &p53, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2_p53 = 95.000000;
float* dev_Mdm2_p53 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_p53, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_p53, &Mdm2_p53, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2_mRNA = 10.000000;
float* dev_Mdm2_mRNA = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_mRNA, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_mRNA, &Mdm2_mRNA, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float p53_mRNA = 10.000000;
float* dev_p53_mRNA = 0;
cudaStatus = cudaMalloc(&dev_p53_mRNA, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_p53_mRNA, &p53_mRNA, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float ATMA = 0.000000;
float* dev_ATMA = 0;
cudaStatus = cudaMalloc(&dev_ATMA, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_ATMA, &ATMA, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float ATMI = 200.000000;
float* dev_ATMI = 0;
cudaStatus = cudaMalloc(&dev_ATMI, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_ATMI, &ATMI, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float p53_P = 0.000000;
float* dev_p53_P = 0;
cudaStatus = cudaMalloc(&dev_p53_P, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_p53_P, &p53_P, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2_P = 0.000000;
float* dev_Mdm2_P = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_P, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_P, &Mdm2_P, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float IR = 0.000000;
float* dev_IR = 0;
cudaStatus = cudaMalloc(&dev_IR, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_IR, &IR, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float ROS = 0.000000;
float* dev_ROS = 0;
cudaStatus = cudaMalloc(&dev_ROS, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_ROS, &ROS, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float damDNA = 0.000000;
float* dev_damDNA = 0;
cudaStatus = cudaMalloc(&dev_damDNA, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_damDNA, &damDNA, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float E1 = 100.000000;
float* dev_E1 = 0;
cudaStatus = cudaMalloc(&dev_E1, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_E1, &E1, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float E2 = 100.000000;
float* dev_E2 = 0;
cudaStatus = cudaMalloc(&dev_E2, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_E2, &E2, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float E1_Ub = 0.000000;
float* dev_E1_Ub = 0;
cudaStatus = cudaMalloc(&dev_E1_Ub, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_E1_Ub, &E1_Ub, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float E2_Ub = 0.000000;
float* dev_E2_Ub = 0;
cudaStatus = cudaMalloc(&dev_E2_Ub, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_E2_Ub, &E2_Ub, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Proteasome = 500.000000;
float* dev_Proteasome = 0;
cudaStatus = cudaMalloc(&dev_Proteasome, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Proteasome, &Proteasome, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Ub = 4000.000000;
float* dev_Ub = 0;
cudaStatus = cudaMalloc(&dev_Ub, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Ub, &Ub, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float p53DUB = 200.000000;
float* dev_p53DUB = 0;
cudaStatus = cudaMalloc(&dev_p53DUB, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_p53DUB, &p53DUB, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2DUB = 200.000000;
float* dev_Mdm2DUB = 0;
cudaStatus = cudaMalloc(&dev_Mdm2DUB, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2DUB, &Mdm2DUB, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float DUB = 200.000000;
float* dev_DUB = 0;
cudaStatus = cudaMalloc(&dev_DUB, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_DUB, &DUB, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2_p53_Ub = 0.000000;
float* dev_Mdm2_p53_Ub = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_p53_Ub, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_p53_Ub, &Mdm2_p53_Ub, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2_p53_Ub2 = 0.000000;
float* dev_Mdm2_p53_Ub2 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_p53_Ub2, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_p53_Ub2, &Mdm2_p53_Ub2, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2_p53_Ub3 = 0.000000;
float* dev_Mdm2_p53_Ub3 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_p53_Ub3, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_p53_Ub3, &Mdm2_p53_Ub3, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2_p53_Ub4 = 0.000000;
float* dev_Mdm2_p53_Ub4 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_p53_Ub4, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_p53_Ub4, &Mdm2_p53_Ub4, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2_P1_p53_Ub4 = 0.000000;
float* dev_Mdm2_P1_p53_Ub4 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_P1_p53_Ub4, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_P1_p53_Ub4, &Mdm2_P1_p53_Ub4, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2_Ub = 0.000000;
float* dev_Mdm2_Ub = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_Ub, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_Ub, &Mdm2_Ub, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2_Ub2 = 0.000000;
float* dev_Mdm2_Ub2 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_Ub2, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_Ub2, &Mdm2_Ub2, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2_Ub3 = 0.000000;
float* dev_Mdm2_Ub3 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_Ub3, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_Ub3, &Mdm2_Ub3, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2_Ub4 = 0.000000;
float* dev_Mdm2_Ub4 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_Ub4, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_Ub4, &Mdm2_Ub4, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2_P_Ub = 0.000000;
float* dev_Mdm2_P_Ub = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_P_Ub, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_P_Ub, &Mdm2_P_Ub, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2_P_Ub2 = 0.000000;
float* dev_Mdm2_P_Ub2 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_P_Ub2, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_P_Ub2, &Mdm2_P_Ub2, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2_P_Ub3 = 0.000000;
float* dev_Mdm2_P_Ub3 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_P_Ub3, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_P_Ub3, &Mdm2_P_Ub3, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2_P_Ub4 = 0.000000;
float* dev_Mdm2_P_Ub4 = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_P_Ub4, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_P_Ub4, &Mdm2_P_Ub4, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float p53_Ub4_Proteasome = 0.000000;
float* dev_p53_Ub4_Proteasome = 0;
cudaStatus = cudaMalloc(&dev_p53_Ub4_Proteasome, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_p53_Ub4_Proteasome, &p53_Ub4_Proteasome, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2_Ub4_Proteasome = 0.000000;
float* dev_Mdm2_Ub4_Proteasome = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_Ub4_Proteasome, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_Ub4_Proteasome, &Mdm2_Ub4_Proteasome, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Mdm2_P_Ub4_Proteasome = 0.000000;
float* dev_Mdm2_P_Ub4_Proteasome = 0;
cudaStatus = cudaMalloc(&dev_Mdm2_P_Ub4_Proteasome, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Mdm2_P_Ub4_Proteasome, &Mdm2_P_Ub4_Proteasome, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float GSK3b = 500.000000;
float* dev_GSK3b = 0;
cudaStatus = cudaMalloc(&dev_GSK3b, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_GSK3b, &GSK3b, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float GSK3b_p53 = 0.000000;
float* dev_GSK3b_p53 = 0;
cudaStatus = cudaMalloc(&dev_GSK3b_p53, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_GSK3b_p53, &GSK3b_p53, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float GSK3b_p53_P = 0.000000;
float* dev_GSK3b_p53_P = 0;
cudaStatus = cudaMalloc(&dev_GSK3b_p53_P, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_GSK3b_p53_P, &GSK3b_p53_P, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Abeta = 0.000000;
float* dev_Abeta = 0;
cudaStatus = cudaMalloc(&dev_Abeta, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Abeta, &Abeta, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float AggAbeta_Proteasome = 0.000000;
float* dev_AggAbeta_Proteasome = 0;
cudaStatus = cudaMalloc(&dev_AggAbeta_Proteasome, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_AggAbeta_Proteasome, &AggAbeta_Proteasome, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float AbetaPlaque = 0.000000;
float* dev_AbetaPlaque = 0;
cudaStatus = cudaMalloc(&dev_AbetaPlaque, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_AbetaPlaque, &AbetaPlaque, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Tau = 0.000000;
float* dev_Tau = 0;
cudaStatus = cudaMalloc(&dev_Tau, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Tau, &Tau, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Tau_P1 = 0.000000;
float* dev_Tau_P1 = 0;
cudaStatus = cudaMalloc(&dev_Tau_P1, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Tau_P1, &Tau_P1, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Tau_P2 = 0.000000;
float* dev_Tau_P2 = 0;
cudaStatus = cudaMalloc(&dev_Tau_P2, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Tau_P2, &Tau_P2, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float MT_Tau = 100.000000;
float* dev_MT_Tau = 0;
cudaStatus = cudaMalloc(&dev_MT_Tau, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_MT_Tau, &MT_Tau, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float AggTau = 0.000000;
float* dev_AggTau = 0;
cudaStatus = cudaMalloc(&dev_AggTau, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_AggTau, &AggTau, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float AggTau_Proteasome = 0.000000;
float* dev_AggTau_Proteasome = 0;
cudaStatus = cudaMalloc(&dev_AggTau_Proteasome, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_AggTau_Proteasome, &AggTau_Proteasome, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Proteasome_Tau = 0.000000;
float* dev_Proteasome_Tau = 0;
cudaStatus = cudaMalloc(&dev_Proteasome_Tau, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Proteasome_Tau, &Proteasome_Tau, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float PP1 = 50.000000;
float* dev_PP1 = 0;
cudaStatus = cudaMalloc(&dev_PP1, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_PP1, &PP1, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float NFT = 0.000000;
float* dev_NFT = 0;
cudaStatus = cudaMalloc(&dev_NFT, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_NFT, &NFT, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float ATP = 10000.000000;
float* dev_ATP = 0;
cudaStatus = cudaMalloc(&dev_ATP, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_ATP, &ATP, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float ADP = 1000.000000;
float* dev_ADP = 0;
cudaStatus = cudaMalloc(&dev_ADP, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_ADP, &ADP, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float AMP = 1000.000000;
float* dev_AMP = 0;
cudaStatus = cudaMalloc(&dev_AMP, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_AMP, &AMP, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float AbetaDimer = 0.000000;
float* dev_AbetaDimer = 0;
cudaStatus = cudaMalloc(&dev_AbetaDimer, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_AbetaDimer, &AbetaDimer, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float AbetaPlaque_GliaA = 0.000000;
float* dev_AbetaPlaque_GliaA = 0;
cudaStatus = cudaMalloc(&dev_AbetaPlaque_GliaA, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_AbetaPlaque_GliaA, &AbetaPlaque_GliaA, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float GliaI = 100.000000;
float* dev_GliaI = 0;
cudaStatus = cudaMalloc(&dev_GliaI, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_GliaI, &GliaI, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float GliaM1 = 0.000000;
float* dev_GliaM1 = 0;
cudaStatus = cudaMalloc(&dev_GliaM1, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_GliaM1, &GliaM1, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float GliaM2 = 0.000000;
float* dev_GliaM2 = 0;
cudaStatus = cudaMalloc(&dev_GliaM2, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_GliaM2, &GliaM2, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float GliaA = 0.000000;
float* dev_GliaA = 0;
cudaStatus = cudaMalloc(&dev_GliaA, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_GliaA, &GliaA, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float antiAb = 0.000000;
float* dev_antiAb = 0;
cudaStatus = cudaMalloc(&dev_antiAb, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_antiAb, &antiAb, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Abeta_antiAb = 0.000000;
float* dev_Abeta_antiAb = 0;
cudaStatus = cudaMalloc(&dev_Abeta_antiAb, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Abeta_antiAb, &Abeta_antiAb, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float AbetaDimer_antiAb = 0.000000;
float* dev_AbetaDimer_antiAb = 0;
cudaStatus = cudaMalloc(&dev_AbetaDimer_antiAb, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_AbetaDimer_antiAb, &AbetaDimer_antiAb, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float degAbetaGlia = 0.000000;
float* dev_degAbetaGlia = 0;
cudaStatus = cudaMalloc(&dev_degAbetaGlia, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_degAbetaGlia, &degAbetaGlia, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float disaggPlaque1 = 0.000000;
float* dev_disaggPlaque1 = 0;
cudaStatus = cudaMalloc(&dev_disaggPlaque1, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_disaggPlaque1, &disaggPlaque1, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float disaggPlaque2 = 0.000000;
float* dev_disaggPlaque2 = 0;
cudaStatus = cudaMalloc(&dev_disaggPlaque2, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_disaggPlaque2, &disaggPlaque2, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Source = 1.000000;
float* dev_Source = 0;
cudaStatus = cudaMalloc(&dev_Source, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Source, &Source, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
float Sink = 1.000000;
float* dev_Sink = 0;
cudaStatus = cudaMalloc(&dev_Sink, sizeof(float));
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
cudaStatus = cudaMemcpy(dev_Sink, &Sink, sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}
simulate<<<1, 112>>>(0.010000, 100000, 0, dev_Mdm2, 0, dev_p53, 0, dev_Mdm2_p53, 0, dev_Mdm2_mRNA, 0, dev_p53_mRNA, 0, dev_ATMA, 0, dev_ATMI, 0, dev_p53_P, 0, dev_Mdm2_P, 0, dev_IR, 0, dev_ROS, 0, dev_damDNA, 0, dev_E1, 0, dev_E2, 0, dev_E1_Ub, 0, dev_E2_Ub, 0, dev_Proteasome, 0, dev_Ub, 0, dev_p53DUB, 0, dev_Mdm2DUB, 0, dev_DUB, 0, dev_Mdm2_p53_Ub, 0, dev_Mdm2_p53_Ub2, 0, dev_Mdm2_p53_Ub3, 0, dev_Mdm2_p53_Ub4, 0, dev_Mdm2_P1_p53_Ub4, 0, dev_Mdm2_Ub, 0, dev_Mdm2_Ub2, 0, dev_Mdm2_Ub3, 0, dev_Mdm2_Ub4, 0, dev_Mdm2_P_Ub, 0, dev_Mdm2_P_Ub2, 0, dev_Mdm2_P_Ub3, 0, dev_Mdm2_P_Ub4, 0, dev_p53_Ub4_Proteasome, 0, dev_Mdm2_Ub4_Proteasome, 0, dev_Mdm2_P_Ub4_Proteasome, 0, dev_GSK3b, 0, dev_GSK3b_p53, 0, dev_GSK3b_p53_P, 0, dev_Abeta, 0, dev_AggAbeta_Proteasome, 0, dev_AbetaPlaque, 0, dev_Tau, 0, dev_Tau_P1, 0, dev_Tau_P2, 0, dev_MT_Tau, 0, dev_AggTau, 0, dev_AggTau_Proteasome, 0, dev_Proteasome_Tau, 0, dev_PP1, 0, dev_NFT, 0, dev_ATP, 0, dev_ADP, 0, dev_AMP, 0, dev_AbetaDimer, 0, dev_AbetaPlaque_GliaA, 0, dev_GliaI, 0, dev_GliaM1, 0, dev_GliaM2, 0, dev_GliaA, 0, dev_antiAb, 0, dev_Abeta_antiAb, 0, dev_AbetaDimer_antiAb, 0, dev_degAbetaGlia, 0, dev_disaggPlaque1, 0, dev_disaggPlaque2, 0, dev_Source, 0, dev_Sink);

cudaStatus = cudaGetLastError(); if (cudaStatus != cudaSuccess) {fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));goto Error;}

cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);goto Error;}cudaStatus = cudaMemcpy(&Mdm2, dev_Mdm2, sizeof(float), cudaMemcpyDeviceToHost);
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

cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);goto Error;}printf("Mdm2 = %f\n", Mdm2);
printf("p53 = %f\n", p53);
printf("Mdm2_p53 = %f\n", Mdm2_p53);
printf("Mdm2_mRNA = %f\n", Mdm2_mRNA);
printf("p53_mRNA = %f\n", p53_mRNA);
printf("ATMA = %f\n", ATMA);
printf("ATMI = %f\n", ATMI);
printf("p53_P = %f\n", p53_P);
printf("Mdm2_P = %f\n", Mdm2_P);
printf("IR = %f\n", IR);
printf("ROS = %f\n", ROS);
printf("damDNA = %f\n", damDNA);
printf("E1 = %f\n", E1);
printf("E2 = %f\n", E2);
printf("E1_Ub = %f\n", E1_Ub);
printf("E2_Ub = %f\n", E2_Ub);
printf("Proteasome = %f\n", Proteasome);
printf("Ub = %f\n", Ub);
printf("p53DUB = %f\n", p53DUB);
printf("Mdm2DUB = %f\n", Mdm2DUB);
printf("DUB = %f\n", DUB);
printf("Mdm2_p53_Ub = %f\n", Mdm2_p53_Ub);
printf("Mdm2_p53_Ub2 = %f\n", Mdm2_p53_Ub2);
printf("Mdm2_p53_Ub3 = %f\n", Mdm2_p53_Ub3);
printf("Mdm2_p53_Ub4 = %f\n", Mdm2_p53_Ub4);
printf("Mdm2_P1_p53_Ub4 = %f\n", Mdm2_P1_p53_Ub4);
printf("Mdm2_Ub = %f\n", Mdm2_Ub);
printf("Mdm2_Ub2 = %f\n", Mdm2_Ub2);
printf("Mdm2_Ub3 = %f\n", Mdm2_Ub3);
printf("Mdm2_Ub4 = %f\n", Mdm2_Ub4);
printf("Mdm2_P_Ub = %f\n", Mdm2_P_Ub);
printf("Mdm2_P_Ub2 = %f\n", Mdm2_P_Ub2);
printf("Mdm2_P_Ub3 = %f\n", Mdm2_P_Ub3);
printf("Mdm2_P_Ub4 = %f\n", Mdm2_P_Ub4);
printf("p53_Ub4_Proteasome = %f\n", p53_Ub4_Proteasome);
printf("Mdm2_Ub4_Proteasome = %f\n", Mdm2_Ub4_Proteasome);
printf("Mdm2_P_Ub4_Proteasome = %f\n", Mdm2_P_Ub4_Proteasome);
printf("GSK3b = %f\n", GSK3b);
printf("GSK3b_p53 = %f\n", GSK3b_p53);
printf("GSK3b_p53_P = %f\n", GSK3b_p53_P);
printf("Abeta = %f\n", Abeta);
printf("AggAbeta_Proteasome = %f\n", AggAbeta_Proteasome);
printf("AbetaPlaque = %f\n", AbetaPlaque);
printf("Tau = %f\n", Tau);
printf("Tau_P1 = %f\n", Tau_P1);
printf("Tau_P2 = %f\n", Tau_P2);
printf("MT_Tau = %f\n", MT_Tau);
printf("AggTau = %f\n", AggTau);
printf("AggTau_Proteasome = %f\n", AggTau_Proteasome);
printf("Proteasome_Tau = %f\n", Proteasome_Tau);
printf("PP1 = %f\n", PP1);
printf("NFT = %f\n", NFT);
printf("ATP = %f\n", ATP);
printf("ADP = %f\n", ADP);
printf("AMP = %f\n", AMP);
printf("AbetaDimer = %f\n", AbetaDimer);
printf("AbetaPlaque_GliaA = %f\n", AbetaPlaque_GliaA);
printf("GliaI = %f\n", GliaI);
printf("GliaM1 = %f\n", GliaM1);
printf("GliaM2 = %f\n", GliaM2);
printf("GliaA = %f\n", GliaA);
printf("antiAb = %f\n", antiAb);
printf("Abeta_antiAb = %f\n", Abeta_antiAb);
printf("AbetaDimer_antiAb = %f\n", AbetaDimer_antiAb);
printf("degAbetaGlia = %f\n", degAbetaGlia);
printf("disaggPlaque1 = %f\n", disaggPlaque1);
printf("disaggPlaque2 = %f\n", disaggPlaque2);
printf("Source = %f\n", Source);
printf("Sink = %f\n", Sink);
Error:
cudaFree(dev_Mdm2);
cudaFree(dev_p53);
cudaFree(dev_Mdm2_p53);
cudaFree(dev_Mdm2_mRNA);
cudaFree(dev_p53_mRNA);
cudaFree(dev_ATMA);
cudaFree(dev_ATMI);
cudaFree(dev_p53_P);
cudaFree(dev_Mdm2_P);
cudaFree(dev_IR);
cudaFree(dev_ROS);
cudaFree(dev_damDNA);
cudaFree(dev_E1);
cudaFree(dev_E2);
cudaFree(dev_E1_Ub);
cudaFree(dev_E2_Ub);
cudaFree(dev_Proteasome);
cudaFree(dev_Ub);
cudaFree(dev_p53DUB);
cudaFree(dev_Mdm2DUB);
cudaFree(dev_DUB);
cudaFree(dev_Mdm2_p53_Ub);
cudaFree(dev_Mdm2_p53_Ub2);
cudaFree(dev_Mdm2_p53_Ub3);
cudaFree(dev_Mdm2_p53_Ub4);
cudaFree(dev_Mdm2_P1_p53_Ub4);
cudaFree(dev_Mdm2_Ub);
cudaFree(dev_Mdm2_Ub2);
cudaFree(dev_Mdm2_Ub3);
cudaFree(dev_Mdm2_Ub4);
cudaFree(dev_Mdm2_P_Ub);
cudaFree(dev_Mdm2_P_Ub2);
cudaFree(dev_Mdm2_P_Ub3);
cudaFree(dev_Mdm2_P_Ub4);
cudaFree(dev_p53_Ub4_Proteasome);
cudaFree(dev_Mdm2_Ub4_Proteasome);
cudaFree(dev_Mdm2_P_Ub4_Proteasome);
cudaFree(dev_GSK3b);
cudaFree(dev_GSK3b_p53);
cudaFree(dev_GSK3b_p53_P);
cudaFree(dev_Abeta);
cudaFree(dev_AggAbeta_Proteasome);
cudaFree(dev_AbetaPlaque);
cudaFree(dev_Tau);
cudaFree(dev_Tau_P1);
cudaFree(dev_Tau_P2);
cudaFree(dev_MT_Tau);
cudaFree(dev_AggTau);
cudaFree(dev_AggTau_Proteasome);
cudaFree(dev_Proteasome_Tau);
cudaFree(dev_PP1);
cudaFree(dev_NFT);
cudaFree(dev_ATP);
cudaFree(dev_ADP);
cudaFree(dev_AMP);
cudaFree(dev_AbetaDimer);
cudaFree(dev_AbetaPlaque_GliaA);
cudaFree(dev_GliaI);
cudaFree(dev_GliaM1);
cudaFree(dev_GliaM2);
cudaFree(dev_GliaA);
cudaFree(dev_antiAb);
cudaFree(dev_Abeta_antiAb);
cudaFree(dev_AbetaDimer_antiAb);
cudaFree(dev_degAbetaGlia);
cudaFree(dev_disaggPlaque1);
cudaFree(dev_disaggPlaque2);
cudaFree(dev_Source);
cudaFree(dev_Sink);

    return 0;
}