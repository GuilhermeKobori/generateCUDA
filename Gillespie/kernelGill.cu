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
#define Mdm2 species[0]
#define Mdm2_id 0
#define p53 species[1]
#define p53_id 1
#define Mdm2_p53 species[2]
#define Mdm2_p53_id 2
#define Mdm2_mRNA species[3]
#define Mdm2_mRNA_id 3
#define p53_mRNA species[4]
#define p53_mRNA_id 4
#define ATMA species[5]
#define ATMA_id 5
#define ATMI species[6]
#define ATMI_id 6
#define p53_P species[7]
#define p53_P_id 7
#define Mdm2_P species[8]
#define Mdm2_P_id 8
#define IR species[9]
#define IR_id 9
#define ROS species[10]
#define ROS_id 10
#define damDNA species[11]
#define damDNA_id 11
#define E1 species[12]
#define E1_id 12
#define E2 species[13]
#define E2_id 13
#define E1_Ub species[14]
#define E1_Ub_id 14
#define E2_Ub species[15]
#define E2_Ub_id 15
#define Proteasome species[16]
#define Proteasome_id 16
#define Ub species[17]
#define Ub_id 17
#define p53DUB species[18]
#define p53DUB_id 18
#define Mdm2DUB species[19]
#define Mdm2DUB_id 19
#define DUB species[20]
#define DUB_id 20
#define Mdm2_p53_Ub species[21]
#define Mdm2_p53_Ub_id 21
#define Mdm2_p53_Ub2 species[22]
#define Mdm2_p53_Ub2_id 22
#define Mdm2_p53_Ub3 species[23]
#define Mdm2_p53_Ub3_id 23
#define Mdm2_p53_Ub4 species[24]
#define Mdm2_p53_Ub4_id 24
#define Mdm2_P1_p53_Ub4 species[25]
#define Mdm2_P1_p53_Ub4_id 25
#define Mdm2_Ub species[26]
#define Mdm2_Ub_id 26
#define Mdm2_Ub2 species[27]
#define Mdm2_Ub2_id 27
#define Mdm2_Ub3 species[28]
#define Mdm2_Ub3_id 28
#define Mdm2_Ub4 species[29]
#define Mdm2_Ub4_id 29
#define Mdm2_P_Ub species[30]
#define Mdm2_P_Ub_id 30
#define Mdm2_P_Ub2 species[31]
#define Mdm2_P_Ub2_id 31
#define Mdm2_P_Ub3 species[32]
#define Mdm2_P_Ub3_id 32
#define Mdm2_P_Ub4 species[33]
#define Mdm2_P_Ub4_id 33
#define p53_Ub4_Proteasome species[34]
#define p53_Ub4_Proteasome_id 34
#define Mdm2_Ub4_Proteasome species[35]
#define Mdm2_Ub4_Proteasome_id 35
#define Mdm2_P_Ub4_Proteasome species[36]
#define Mdm2_P_Ub4_Proteasome_id 36
#define GSK3b species[37]
#define GSK3b_id 37
#define GSK3b_p53 species[38]
#define GSK3b_p53_id 38
#define GSK3b_p53_P species[39]
#define GSK3b_p53_P_id 39
#define Abeta species[40]
#define Abeta_id 40
#define AggAbeta_Proteasome species[41]
#define AggAbeta_Proteasome_id 41
#define AbetaPlaque species[42]
#define AbetaPlaque_id 42
#define Tau species[43]
#define Tau_id 43
#define Tau_P1 species[44]
#define Tau_P1_id 44
#define Tau_P2 species[45]
#define Tau_P2_id 45
#define MT_Tau species[46]
#define MT_Tau_id 46
#define AggTau species[47]
#define AggTau_id 47
#define AggTau_Proteasome species[48]
#define AggTau_Proteasome_id 48
#define Proteasome_Tau species[49]
#define Proteasome_Tau_id 49
#define PP1 species[50]
#define PP1_id 50
#define NFT species[51]
#define NFT_id 51
#define ATP species[52]
#define ATP_id 52
#define ADP species[53]
#define ADP_id 53
#define AMP species[54]
#define AMP_id 54
#define AbetaDimer species[55]
#define AbetaDimer_id 55
#define AbetaPlaque_GliaA species[56]
#define AbetaPlaque_GliaA_id 56
#define GliaI species[57]
#define GliaI_id 57
#define GliaM1 species[58]
#define GliaM1_id 58
#define GliaM2 species[59]
#define GliaM2_id 59
#define GliaA species[60]
#define GliaA_id 60
#define antiAb species[61]
#define antiAb_id 61
#define Abeta_antiAb species[62]
#define Abeta_antiAb_id 62
#define AbetaDimer_antiAb species[63]
#define AbetaDimer_antiAb_id 63
#define degAbetaGlia species[64]
#define degAbetaGlia_id 64
#define disaggPlaque1 species[65]
#define disaggPlaque1_id 65
#define disaggPlaque2 species[66]
#define disaggPlaque2_id 66
#define Source species[67]
#define Source_id 67
#define Sink species[68]
#define Sink_id 68
#define cell 1.0000000000
#define ksynp53mRNA 0.0010000000
#define kdegp53mRNA 0.0001000000
#define ksynMdm2mRNA 0.0005000000
#define kdegMdm2mRNA 0.0005000000
#define ksynMdm2mRNAGSK3bp53 0.0007000000
#define ksynp53 0.0070000000
#define kdegp53 0.0050000000
#define kbinMdm2p53 0.0011550000
#define krelMdm2p53 0.0000115500
#define kbinGSK3bp53 0.0000020000
#define krelGSK3bp53 0.0020000000
#define ksynMdm2 0.0004950000
#define kdegMdm2 0.0100000000
#define kbinE1Ub 0.0002000000
#define kbinE2Ub 0.0010000000
#define kp53Ub 0.0000500000
#define kp53PolyUb 0.0100000000
#define kbinProt 0.0000020000
#define kactDUBp53 0.0000001000
#define kactDUBProtp53 0.0001000000
#define kactDUBMdm2 0.0000001000
#define kMdm2Ub 0.0000045600
#define kMdm2PUb 0.0000068400
#define kMdm2PolyUb 0.0045600000
#define kdam 0.0800000000
#define krepair 0.0000200000
#define kactATM 0.0001000000
#define kinactATM 0.0005000000
#define kphosp53 0.0002000000
#define kdephosp53 0.5000000000
#define kphosMdm2 2.0000000000
#define kdephosMdm2 0.5000000000
#define kphosMdm2GSK3b 0.0050000000
#define kphosMdm2GSK3bp53 0.5000000000
#define kphospTauGSK3bp53 0.1000000000
#define kphospTauGSK3b 0.0002000000
#define kdephospTau 0.0100000000
#define kbinMTTau 0.1000000000
#define krelMTTau 0.0001000000
#define ksynTau 0.0000800000
#define kbinTauProt 0.0000001925
#define kdegTau20SProt 0.0100000000
#define kaggTau 0.0000000100
#define kaggTauP1 0.0000000100
#define kaggTauP2 0.0000001000
#define ktangfor 0.0010000000
#define kinhibprot 0.0000001000
#define ksynp53mRNAAbeta 0.0000100000
#define kdamROS 0.0000100000
#define kgenROSAbeta 0.0000200000
#define kgenROSPlaque 0.0000100000
#define kgenROSGlia 0.0000100000
#define kproteff 1.0000000000
#define kremROS 0.0000700000
#define kprodAbeta 0.0000186000
#define kprodAbeta2 0.0000186000
#define kdegAbeta 0.0000150000
#define kaggAbeta 0.0000030000
#define kdisaggAbeta 0.0000010000
#define kdisaggAbeta1 0.0002000000
#define kdisaggAbeta2 0.0000010000
#define kdegAbetaGlia 0.0050000000
#define kpf 0.2000000000
#define kpg 0.1500000000
#define kpghalf 10.0000000000
#define kactglia1 0.0000006000
#define kactglia2 0.0000006000
#define kinactglia1 0.0000050000
#define kinactglia2 0.0000050000
#define kbinAbetaGlia 0.0000100000
#define krelAbetaGlia 0.0000500000
#define kdegAntiAb 0.0000027500
#define kbinAbantiAb 0.0000010000

__global__
void simulate(int numberOfExecutions, float* output, curandState *state, float step, float endTime, float segmentSize, float* Mdm2_aux, float* Mdm2_global, float* p53_aux, float* p53_global, float* Mdm2_p53_aux, float* Mdm2_p53_global, float* Mdm2_mRNA_aux, float* Mdm2_mRNA_global, float* p53_mRNA_aux, float* p53_mRNA_global, float* ATMA_aux, float* ATMA_global, float* ATMI_aux, float* ATMI_global, float* p53_P_aux, float* p53_P_global, float* Mdm2_P_aux, float* Mdm2_P_global, float* IR_aux, float* IR_global, float* ROS_aux, float* ROS_global, float* damDNA_aux, float* damDNA_global, float* E1_aux, float* E1_global, float* E2_aux, float* E2_global, float* E1_Ub_aux, float* E1_Ub_global, float* E2_Ub_aux, float* E2_Ub_global, float* Proteasome_aux, float* Proteasome_global, float* Ub_aux, float* Ub_global, float* p53DUB_aux, float* p53DUB_global, float* Mdm2DUB_aux, float* Mdm2DUB_global, float* DUB_aux, float* DUB_global, float* Mdm2_p53_Ub_aux, float* Mdm2_p53_Ub_global, float* Mdm2_p53_Ub2_aux, float* Mdm2_p53_Ub2_global, float* Mdm2_p53_Ub3_aux, float* Mdm2_p53_Ub3_global, float* Mdm2_p53_Ub4_aux, float* Mdm2_p53_Ub4_global, float* Mdm2_P1_p53_Ub4_aux, float* Mdm2_P1_p53_Ub4_global, float* Mdm2_Ub_aux, float* Mdm2_Ub_global, float* Mdm2_Ub2_aux, float* Mdm2_Ub2_global, float* Mdm2_Ub3_aux, float* Mdm2_Ub3_global, float* Mdm2_Ub4_aux, float* Mdm2_Ub4_global, float* Mdm2_P_Ub_aux, float* Mdm2_P_Ub_global, float* Mdm2_P_Ub2_aux, float* Mdm2_P_Ub2_global, float* Mdm2_P_Ub3_aux, float* Mdm2_P_Ub3_global, float* Mdm2_P_Ub4_aux, float* Mdm2_P_Ub4_global, float* p53_Ub4_Proteasome_aux, float* p53_Ub4_Proteasome_global, float* Mdm2_Ub4_Proteasome_aux, float* Mdm2_Ub4_Proteasome_global, float* Mdm2_P_Ub4_Proteasome_aux, float* Mdm2_P_Ub4_Proteasome_global, float* GSK3b_aux, float* GSK3b_global, float* GSK3b_p53_aux, float* GSK3b_p53_global, float* GSK3b_p53_P_aux, float* GSK3b_p53_P_global, float* Abeta_aux, float* Abeta_global, float* AggAbeta_Proteasome_aux, float* AggAbeta_Proteasome_global, float* AbetaPlaque_aux, float* AbetaPlaque_global, float* Tau_aux, float* Tau_global, float* Tau_P1_aux, float* Tau_P1_global, float* Tau_P2_aux, float* Tau_P2_global, float* MT_Tau_aux, float* MT_Tau_global, float* AggTau_aux, float* AggTau_global, float* AggTau_Proteasome_aux, float* AggTau_Proteasome_global, float* Proteasome_Tau_aux, float* Proteasome_Tau_global, float* PP1_aux, float* PP1_global, float* NFT_aux, float* NFT_global, float* ATP_aux, float* ATP_global, float* ADP_aux, float* ADP_global, float* AMP_aux, float* AMP_global, float* AbetaDimer_aux, float* AbetaDimer_global, float* AbetaPlaque_GliaA_aux, float* AbetaPlaque_GliaA_global, float* GliaI_aux, float* GliaI_global, float* GliaM1_aux, float* GliaM1_global, float* GliaM2_aux, float* GliaM2_global, float* GliaA_aux, float* GliaA_global, float* antiAb_aux, float* antiAb_global, float* Abeta_antiAb_aux, float* Abeta_antiAb_global, float* AbetaDimer_antiAb_aux, float* AbetaDimer_antiAb_global, float* degAbetaGlia_aux, float* degAbetaGlia_global, float* disaggPlaque1_aux, float* disaggPlaque1_global, float* disaggPlaque2_aux, float* disaggPlaque2_global, float* Source_aux, float* Source_global, float* Sink_aux, float* Sink_global) {
	int reaction, stepCount = 0;
	int indexMin, indexMax;
	float time = numberOfExecutions * segmentSize;
	float sum_p, timeStep, random;
	float cummulative_p[112];
	int triggerEvent0 = 0;
	if (time >= 345600) { triggerEvent0 = 1; }
	float species[69];
	if (numberOfExecutions == 0) {
		species[0] = *Mdm2_aux;
	}
	else {
		species[0] = Mdm2_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[1] = *p53_aux;
	}
	else {
		species[1] = p53_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[2] = *Mdm2_p53_aux;
	}
	else {
		species[2] = Mdm2_p53_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[3] = *Mdm2_mRNA_aux;
	}
	else {
		species[3] = Mdm2_mRNA_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[4] = *p53_mRNA_aux;
	}
	else {
		species[4] = p53_mRNA_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[5] = *ATMA_aux;
	}
	else {
		species[5] = ATMA_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[6] = *ATMI_aux;
	}
	else {
		species[6] = ATMI_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[7] = *p53_P_aux;
	}
	else {
		species[7] = p53_P_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[8] = *Mdm2_P_aux;
	}
	else {
		species[8] = Mdm2_P_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[9] = *IR_aux;
	}
	else {
		species[9] = IR_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[10] = *ROS_aux;
	}
	else {
		species[10] = ROS_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[11] = *damDNA_aux;
	}
	else {
		species[11] = damDNA_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[12] = *E1_aux;
	}
	else {
		species[12] = E1_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[13] = *E2_aux;
	}
	else {
		species[13] = E2_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[14] = *E1_Ub_aux;
	}
	else {
		species[14] = E1_Ub_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[15] = *E2_Ub_aux;
	}
	else {
		species[15] = E2_Ub_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[16] = *Proteasome_aux;
	}
	else {
		species[16] = Proteasome_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[17] = *Ub_aux;
	}
	else {
		species[17] = Ub_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[18] = *p53DUB_aux;
	}
	else {
		species[18] = p53DUB_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[19] = *Mdm2DUB_aux;
	}
	else {
		species[19] = Mdm2DUB_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[20] = *DUB_aux;
	}
	else {
		species[20] = DUB_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[21] = *Mdm2_p53_Ub_aux;
	}
	else {
		species[21] = Mdm2_p53_Ub_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[22] = *Mdm2_p53_Ub2_aux;
	}
	else {
		species[22] = Mdm2_p53_Ub2_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[23] = *Mdm2_p53_Ub3_aux;
	}
	else {
		species[23] = Mdm2_p53_Ub3_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[24] = *Mdm2_p53_Ub4_aux;
	}
	else {
		species[24] = Mdm2_p53_Ub4_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[25] = *Mdm2_P1_p53_Ub4_aux;
	}
	else {
		species[25] = Mdm2_P1_p53_Ub4_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[26] = *Mdm2_Ub_aux;
	}
	else {
		species[26] = Mdm2_Ub_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[27] = *Mdm2_Ub2_aux;
	}
	else {
		species[27] = Mdm2_Ub2_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[28] = *Mdm2_Ub3_aux;
	}
	else {
		species[28] = Mdm2_Ub3_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[29] = *Mdm2_Ub4_aux;
	}
	else {
		species[29] = Mdm2_Ub4_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[30] = *Mdm2_P_Ub_aux;
	}
	else {
		species[30] = Mdm2_P_Ub_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[31] = *Mdm2_P_Ub2_aux;
	}
	else {
		species[31] = Mdm2_P_Ub2_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[32] = *Mdm2_P_Ub3_aux;
	}
	else {
		species[32] = Mdm2_P_Ub3_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[33] = *Mdm2_P_Ub4_aux;
	}
	else {
		species[33] = Mdm2_P_Ub4_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[34] = *p53_Ub4_Proteasome_aux;
	}
	else {
		species[34] = p53_Ub4_Proteasome_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[35] = *Mdm2_Ub4_Proteasome_aux;
	}
	else {
		species[35] = Mdm2_Ub4_Proteasome_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[36] = *Mdm2_P_Ub4_Proteasome_aux;
	}
	else {
		species[36] = Mdm2_P_Ub4_Proteasome_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[37] = *GSK3b_aux;
	}
	else {
		species[37] = GSK3b_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[38] = *GSK3b_p53_aux;
	}
	else {
		species[38] = GSK3b_p53_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[39] = *GSK3b_p53_P_aux;
	}
	else {
		species[39] = GSK3b_p53_P_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[40] = *Abeta_aux;
	}
	else {
		species[40] = Abeta_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[41] = *AggAbeta_Proteasome_aux;
	}
	else {
		species[41] = AggAbeta_Proteasome_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[42] = *AbetaPlaque_aux;
	}
	else {
		species[42] = AbetaPlaque_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[43] = *Tau_aux;
	}
	else {
		species[43] = Tau_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[44] = *Tau_P1_aux;
	}
	else {
		species[44] = Tau_P1_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[45] = *Tau_P2_aux;
	}
	else {
		species[45] = Tau_P2_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[46] = *MT_Tau_aux;
	}
	else {
		species[46] = MT_Tau_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[47] = *AggTau_aux;
	}
	else {
		species[47] = AggTau_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[48] = *AggTau_Proteasome_aux;
	}
	else {
		species[48] = AggTau_Proteasome_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[49] = *Proteasome_Tau_aux;
	}
	else {
		species[49] = Proteasome_Tau_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[50] = *PP1_aux;
	}
	else {
		species[50] = PP1_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[51] = *NFT_aux;
	}
	else {
		species[51] = NFT_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[52] = *ATP_aux;
	}
	else {
		species[52] = ATP_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[53] = *ADP_aux;
	}
	else {
		species[53] = ADP_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[54] = *AMP_aux;
	}
	else {
		species[54] = AMP_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[55] = *AbetaDimer_aux;
	}
	else {
		species[55] = AbetaDimer_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[56] = *AbetaPlaque_GliaA_aux;
	}
	else {
		species[56] = AbetaPlaque_GliaA_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[57] = *GliaI_aux;
	}
	else {
		species[57] = GliaI_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[58] = *GliaM1_aux;
	}
	else {
		species[58] = GliaM1_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[59] = *GliaM2_aux;
	}
	else {
		species[59] = GliaM2_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[60] = *GliaA_aux;
	}
	else {
		species[60] = GliaA_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[61] = *antiAb_aux;
	}
	else {
		species[61] = antiAb_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[62] = *Abeta_antiAb_aux;
	}
	else {
		species[62] = Abeta_antiAb_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[63] = *AbetaDimer_antiAb_aux;
	}
	else {
		species[63] = AbetaDimer_antiAb_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[64] = *degAbetaGlia_aux;
	}
	else {
		species[64] = degAbetaGlia_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[65] = *disaggPlaque1_aux;
	}
	else {
		species[65] = disaggPlaque1_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[66] = *disaggPlaque2_aux;
	}
	else {
		species[66] = disaggPlaque2_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[67] = *Source_aux;
	}
	else {
		species[67] = Source_global[threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[68] = *Sink_aux;
	}
	else {
		species[68] = Sink_global[threadIdx.x];
	}
	int reactionsSpecies[112][5];
	int reactionsValues[112][5];
	for (int i = 0; i < 112; i++) {
		for (int j = 0; j < 5; j++) {
			reactionsSpecies[i][j] = -1;
			reactionsValues[i][j] = 0;
		}
	}
	reactionsSpecies[0][0] = p53_mRNA_id;
	reactionsValues[0][0] = 1.0000000000;
	reactionsSpecies[1][0] = p53_mRNA_id;
	reactionsValues[1][0] = -1.0000000000;
	reactionsSpecies[2][0] = Mdm2_mRNA_id;
	reactionsValues[2][0] = -1.0000000000;
	reactionsSpecies[2][1] = Mdm2_mRNA_id;
	reactionsValues[2][1] = 1.0000000000;
	reactionsSpecies[2][2] = Mdm2_id;
	reactionsValues[2][2] = 1.0000000000;
	reactionsSpecies[3][0] = p53_id;
	reactionsValues[3][0] = -1.0000000000;
	reactionsSpecies[3][1] = p53_id;
	reactionsValues[3][1] = 1.0000000000;
	reactionsSpecies[3][2] = Mdm2_mRNA_id;
	reactionsValues[3][2] = 1.0000000000;
	reactionsSpecies[4][0] = p53_P_id;
	reactionsValues[4][0] = -1.0000000000;
	reactionsSpecies[4][1] = p53_P_id;
	reactionsValues[4][1] = 1.0000000000;
	reactionsSpecies[4][2] = Mdm2_mRNA_id;
	reactionsValues[4][2] = 1.0000000000;
	reactionsSpecies[5][0] = GSK3b_p53_id;
	reactionsValues[5][0] = -1.0000000000;
	reactionsSpecies[5][1] = GSK3b_p53_id;
	reactionsValues[5][1] = 1.0000000000;
	reactionsSpecies[5][2] = Mdm2_mRNA_id;
	reactionsValues[5][2] = 1.0000000000;
	reactionsSpecies[6][0] = GSK3b_p53_P_id;
	reactionsValues[6][0] = -1.0000000000;
	reactionsSpecies[6][1] = GSK3b_p53_P_id;
	reactionsValues[6][1] = 1.0000000000;
	reactionsSpecies[6][2] = Mdm2_mRNA_id;
	reactionsValues[6][2] = 1.0000000000;
	reactionsSpecies[7][0] = Mdm2_mRNA_id;
	reactionsValues[7][0] = -1.0000000000;
	reactionsSpecies[8][0] = p53_id;
	reactionsValues[8][0] = -1.0000000000;
	reactionsSpecies[8][1] = Mdm2_id;
	reactionsValues[8][1] = -1.0000000000;
	reactionsSpecies[8][2] = Mdm2_p53_id;
	reactionsValues[8][2] = 1.0000000000;
	reactionsSpecies[9][0] = Mdm2_p53_id;
	reactionsValues[9][0] = -1.0000000000;
	reactionsSpecies[9][1] = p53_id;
	reactionsValues[9][1] = 1.0000000000;
	reactionsSpecies[9][2] = Mdm2_id;
	reactionsValues[9][2] = 1.0000000000;
	reactionsSpecies[10][0] = GSK3b_id;
	reactionsValues[10][0] = -1.0000000000;
	reactionsSpecies[10][1] = p53_id;
	reactionsValues[10][1] = -1.0000000000;
	reactionsSpecies[10][2] = GSK3b_p53_id;
	reactionsValues[10][2] = 1.0000000000;
	reactionsSpecies[11][0] = GSK3b_p53_id;
	reactionsValues[11][0] = -1.0000000000;
	reactionsSpecies[11][1] = GSK3b_id;
	reactionsValues[11][1] = 1.0000000000;
	reactionsSpecies[11][2] = p53_id;
	reactionsValues[11][2] = 1.0000000000;
	reactionsSpecies[12][0] = GSK3b_id;
	reactionsValues[12][0] = -1.0000000000;
	reactionsSpecies[12][1] = p53_P_id;
	reactionsValues[12][1] = -1.0000000000;
	reactionsSpecies[12][2] = GSK3b_p53_P_id;
	reactionsValues[12][2] = 1.0000000000;
	reactionsSpecies[13][0] = GSK3b_p53_P_id;
	reactionsValues[13][0] = -1.0000000000;
	reactionsSpecies[13][1] = GSK3b_id;
	reactionsValues[13][1] = 1.0000000000;
	reactionsSpecies[13][2] = p53_P_id;
	reactionsValues[13][2] = 1.0000000000;
	reactionsSpecies[14][0] = E1_id;
	reactionsValues[14][0] = -1.0000000000;
	reactionsSpecies[14][1] = Ub_id;
	reactionsValues[14][1] = -1.0000000000;
	reactionsSpecies[14][2] = E1_Ub_id;
	reactionsValues[14][2] = 1.0000000000;
	reactionsSpecies[15][0] = E2_id;
	reactionsValues[15][0] = -1.0000000000;
	reactionsSpecies[15][1] = E1_Ub_id;
	reactionsValues[15][1] = -1.0000000000;
	reactionsSpecies[15][2] = E2_Ub_id;
	reactionsValues[15][2] = 1.0000000000;
	reactionsSpecies[15][3] = E1_id;
	reactionsValues[15][3] = 1.0000000000;
	reactionsSpecies[16][0] = Mdm2_id;
	reactionsValues[16][0] = -1.0000000000;
	reactionsSpecies[16][1] = E2_Ub_id;
	reactionsValues[16][1] = -1.0000000000;
	reactionsSpecies[16][2] = Mdm2_Ub_id;
	reactionsValues[16][2] = 1.0000000000;
	reactionsSpecies[16][3] = E2_id;
	reactionsValues[16][3] = 1.0000000000;
	reactionsSpecies[17][0] = Mdm2_Ub_id;
	reactionsValues[17][0] = -1.0000000000;
	reactionsSpecies[17][1] = E2_Ub_id;
	reactionsValues[17][1] = -1.0000000000;
	reactionsSpecies[17][2] = Mdm2_Ub2_id;
	reactionsValues[17][2] = 1.0000000000;
	reactionsSpecies[17][3] = E2_id;
	reactionsValues[17][3] = 1.0000000000;
	reactionsSpecies[18][0] = Mdm2_Ub2_id;
	reactionsValues[18][0] = -1.0000000000;
	reactionsSpecies[18][1] = E2_Ub_id;
	reactionsValues[18][1] = -1.0000000000;
	reactionsSpecies[18][2] = Mdm2_Ub3_id;
	reactionsValues[18][2] = 1.0000000000;
	reactionsSpecies[18][3] = E2_id;
	reactionsValues[18][3] = 1.0000000000;
	reactionsSpecies[19][0] = Mdm2_Ub3_id;
	reactionsValues[19][0] = -1.0000000000;
	reactionsSpecies[19][1] = E2_Ub_id;
	reactionsValues[19][1] = -1.0000000000;
	reactionsSpecies[19][2] = Mdm2_Ub4_id;
	reactionsValues[19][2] = 1.0000000000;
	reactionsSpecies[19][3] = E2_id;
	reactionsValues[19][3] = 1.0000000000;
	reactionsSpecies[20][0] = Mdm2_Ub4_id;
	reactionsValues[20][0] = -1.0000000000;
	reactionsSpecies[20][1] = Mdm2DUB_id;
	reactionsValues[20][1] = -1.0000000000;
	reactionsSpecies[20][2] = Mdm2_Ub3_id;
	reactionsValues[20][2] = 1.0000000000;
	reactionsSpecies[20][3] = Mdm2DUB_id;
	reactionsValues[20][3] = 1.0000000000;
	reactionsSpecies[20][4] = Ub_id;
	reactionsValues[20][4] = 1.0000000000;
	reactionsSpecies[21][0] = Mdm2_Ub3_id;
	reactionsValues[21][0] = -1.0000000000;
	reactionsSpecies[21][1] = Mdm2DUB_id;
	reactionsValues[21][1] = -1.0000000000;
	reactionsSpecies[21][2] = Mdm2_Ub2_id;
	reactionsValues[21][2] = 1.0000000000;
	reactionsSpecies[21][3] = Mdm2DUB_id;
	reactionsValues[21][3] = 1.0000000000;
	reactionsSpecies[21][4] = Ub_id;
	reactionsValues[21][4] = 1.0000000000;
	reactionsSpecies[22][0] = Mdm2_Ub2_id;
	reactionsValues[22][0] = -1.0000000000;
	reactionsSpecies[22][1] = Mdm2DUB_id;
	reactionsValues[22][1] = -1.0000000000;
	reactionsSpecies[22][2] = Mdm2_Ub_id;
	reactionsValues[22][2] = 1.0000000000;
	reactionsSpecies[22][3] = Mdm2DUB_id;
	reactionsValues[22][3] = 1.0000000000;
	reactionsSpecies[22][4] = Ub_id;
	reactionsValues[22][4] = 1.0000000000;
	reactionsSpecies[23][0] = Mdm2_Ub_id;
	reactionsValues[23][0] = -1.0000000000;
	reactionsSpecies[23][1] = Mdm2DUB_id;
	reactionsValues[23][1] = -1.0000000000;
	reactionsSpecies[23][2] = Mdm2_id;
	reactionsValues[23][2] = 1.0000000000;
	reactionsSpecies[23][3] = Mdm2DUB_id;
	reactionsValues[23][3] = 1.0000000000;
	reactionsSpecies[23][4] = Ub_id;
	reactionsValues[23][4] = 1.0000000000;
	reactionsSpecies[24][0] = Mdm2_Ub4_id;
	reactionsValues[24][0] = -1.0000000000;
	reactionsSpecies[24][1] = Proteasome_id;
	reactionsValues[24][1] = -1.0000000000;
	reactionsSpecies[24][2] = Mdm2_Ub4_Proteasome_id;
	reactionsValues[24][2] = 1.0000000000;
	reactionsSpecies[25][0] = Mdm2_Ub4_Proteasome_id;
	reactionsValues[25][0] = -1.0000000000;
	reactionsSpecies[25][1] = Proteasome_id;
	reactionsValues[25][1] = 1.0000000000;
	reactionsSpecies[25][2] = Ub_id;
	reactionsValues[25][2] = 4.0000000000;
	reactionsSpecies[26][0] = p53_mRNA_id;
	reactionsValues[26][0] = -1.0000000000;
	reactionsSpecies[26][1] = p53_id;
	reactionsValues[26][1] = 1.0000000000;
	reactionsSpecies[26][2] = p53_mRNA_id;
	reactionsValues[26][2] = 1.0000000000;
	reactionsSpecies[27][0] = E2_Ub_id;
	reactionsValues[27][0] = -1.0000000000;
	reactionsSpecies[27][1] = Mdm2_p53_id;
	reactionsValues[27][1] = -1.0000000000;
	reactionsSpecies[27][2] = Mdm2_p53_Ub_id;
	reactionsValues[27][2] = 1.0000000000;
	reactionsSpecies[27][3] = E2_id;
	reactionsValues[27][3] = 1.0000000000;
	reactionsSpecies[28][0] = Mdm2_p53_Ub_id;
	reactionsValues[28][0] = -1.0000000000;
	reactionsSpecies[28][1] = E2_Ub_id;
	reactionsValues[28][1] = -1.0000000000;
	reactionsSpecies[28][2] = Mdm2_p53_Ub2_id;
	reactionsValues[28][2] = 1.0000000000;
	reactionsSpecies[28][3] = E2_id;
	reactionsValues[28][3] = 1.0000000000;
	reactionsSpecies[29][0] = Mdm2_p53_Ub2_id;
	reactionsValues[29][0] = -1.0000000000;
	reactionsSpecies[29][1] = E2_Ub_id;
	reactionsValues[29][1] = -1.0000000000;
	reactionsSpecies[29][2] = Mdm2_p53_Ub3_id;
	reactionsValues[29][2] = 1.0000000000;
	reactionsSpecies[29][3] = E2_id;
	reactionsValues[29][3] = 1.0000000000;
	reactionsSpecies[30][0] = Mdm2_p53_Ub3_id;
	reactionsValues[30][0] = -1.0000000000;
	reactionsSpecies[30][1] = E2_Ub_id;
	reactionsValues[30][1] = -1.0000000000;
	reactionsSpecies[30][2] = Mdm2_p53_Ub4_id;
	reactionsValues[30][2] = 1.0000000000;
	reactionsSpecies[30][3] = E2_id;
	reactionsValues[30][3] = 1.0000000000;
	reactionsSpecies[31][0] = Mdm2_p53_Ub4_id;
	reactionsValues[31][0] = -1.0000000000;
	reactionsSpecies[31][1] = p53DUB_id;
	reactionsValues[31][1] = -1.0000000000;
	reactionsSpecies[31][2] = Mdm2_p53_Ub3_id;
	reactionsValues[31][2] = 1.0000000000;
	reactionsSpecies[31][3] = p53DUB_id;
	reactionsValues[31][3] = 1.0000000000;
	reactionsSpecies[31][4] = Ub_id;
	reactionsValues[31][4] = 1.0000000000;
	reactionsSpecies[32][0] = Mdm2_p53_Ub3_id;
	reactionsValues[32][0] = -1.0000000000;
	reactionsSpecies[32][1] = p53DUB_id;
	reactionsValues[32][1] = -1.0000000000;
	reactionsSpecies[32][2] = Mdm2_p53_Ub2_id;
	reactionsValues[32][2] = 1.0000000000;
	reactionsSpecies[32][3] = p53DUB_id;
	reactionsValues[32][3] = 1.0000000000;
	reactionsSpecies[32][4] = Ub_id;
	reactionsValues[32][4] = 1.0000000000;
	reactionsSpecies[33][0] = Mdm2_p53_Ub2_id;
	reactionsValues[33][0] = -1.0000000000;
	reactionsSpecies[33][1] = p53DUB_id;
	reactionsValues[33][1] = -1.0000000000;
	reactionsSpecies[33][2] = Mdm2_p53_Ub_id;
	reactionsValues[33][2] = 1.0000000000;
	reactionsSpecies[33][3] = p53DUB_id;
	reactionsValues[33][3] = 1.0000000000;
	reactionsSpecies[33][4] = Ub_id;
	reactionsValues[33][4] = 1.0000000000;
	reactionsSpecies[34][0] = Mdm2_p53_Ub_id;
	reactionsValues[34][0] = -1.0000000000;
	reactionsSpecies[34][1] = p53DUB_id;
	reactionsValues[34][1] = -1.0000000000;
	reactionsSpecies[34][2] = Mdm2_p53_id;
	reactionsValues[34][2] = 1.0000000000;
	reactionsSpecies[34][3] = p53DUB_id;
	reactionsValues[34][3] = 1.0000000000;
	reactionsSpecies[34][4] = Ub_id;
	reactionsValues[34][4] = 1.0000000000;
	reactionsSpecies[35][0] = Mdm2_p53_Ub4_id;
	reactionsValues[35][0] = -1.0000000000;
	reactionsSpecies[35][1] = GSK3b_id;
	reactionsValues[35][1] = -1.0000000000;
	reactionsSpecies[35][2] = Mdm2_P1_p53_Ub4_id;
	reactionsValues[35][2] = 1.0000000000;
	reactionsSpecies[35][3] = GSK3b_id;
	reactionsValues[35][3] = 1.0000000000;
	reactionsSpecies[36][0] = Mdm2_p53_Ub4_id;
	reactionsValues[36][0] = -1.0000000000;
	reactionsSpecies[36][1] = GSK3b_p53_id;
	reactionsValues[36][1] = -1.0000000000;
	reactionsSpecies[36][2] = Mdm2_P1_p53_Ub4_id;
	reactionsValues[36][2] = 1.0000000000;
	reactionsSpecies[36][3] = GSK3b_p53_id;
	reactionsValues[36][3] = 1.0000000000;
	reactionsSpecies[37][0] = Mdm2_p53_Ub4_id;
	reactionsValues[37][0] = -1.0000000000;
	reactionsSpecies[37][1] = GSK3b_p53_P_id;
	reactionsValues[37][1] = -1.0000000000;
	reactionsSpecies[37][2] = Mdm2_P1_p53_Ub4_id;
	reactionsValues[37][2] = 1.0000000000;
	reactionsSpecies[37][3] = GSK3b_p53_P_id;
	reactionsValues[37][3] = 1.0000000000;
	reactionsSpecies[38][0] = Mdm2_P1_p53_Ub4_id;
	reactionsValues[38][0] = -1.0000000000;
	reactionsSpecies[38][1] = Proteasome_id;
	reactionsValues[38][1] = -1.0000000000;
	reactionsSpecies[38][2] = p53_Ub4_Proteasome_id;
	reactionsValues[38][2] = 1.0000000000;
	reactionsSpecies[38][3] = Mdm2_id;
	reactionsValues[38][3] = 1.0000000000;
	reactionsSpecies[39][0] = p53_Ub4_Proteasome_id;
	reactionsValues[39][0] = -1.0000000000;
	reactionsSpecies[39][1] = Ub_id;
	reactionsValues[39][1] = 4.0000000000;
	reactionsSpecies[39][2] = Proteasome_id;
	reactionsValues[39][2] = 1.0000000000;
	reactionsSpecies[40][0] = Tau_id;
	reactionsValues[40][0] = -1.0000000000;
	reactionsSpecies[40][1] = MT_Tau_id;
	reactionsValues[40][1] = 1.0000000000;
	reactionsSpecies[41][0] = MT_Tau_id;
	reactionsValues[41][0] = -1.0000000000;
	reactionsSpecies[41][1] = Tau_id;
	reactionsValues[41][1] = 1.0000000000;
	reactionsSpecies[42][0] = GSK3b_p53_id;
	reactionsValues[42][0] = -1.0000000000;
	reactionsSpecies[42][1] = Tau_id;
	reactionsValues[42][1] = -1.0000000000;
	reactionsSpecies[42][2] = GSK3b_p53_id;
	reactionsValues[42][2] = 1.0000000000;
	reactionsSpecies[42][3] = Tau_P1_id;
	reactionsValues[42][3] = 1.0000000000;
	reactionsSpecies[43][0] = GSK3b_p53_id;
	reactionsValues[43][0] = -1.0000000000;
	reactionsSpecies[43][1] = Tau_P1_id;
	reactionsValues[43][1] = -1.0000000000;
	reactionsSpecies[43][2] = GSK3b_p53_id;
	reactionsValues[43][2] = 1.0000000000;
	reactionsSpecies[43][3] = Tau_P2_id;
	reactionsValues[43][3] = 1.0000000000;
	reactionsSpecies[44][0] = GSK3b_p53_P_id;
	reactionsValues[44][0] = -1.0000000000;
	reactionsSpecies[44][1] = Tau_id;
	reactionsValues[44][1] = -1.0000000000;
	reactionsSpecies[44][2] = GSK3b_p53_P_id;
	reactionsValues[44][2] = 1.0000000000;
	reactionsSpecies[44][3] = Tau_P1_id;
	reactionsValues[44][3] = 1.0000000000;
	reactionsSpecies[45][0] = GSK3b_p53_P_id;
	reactionsValues[45][0] = -1.0000000000;
	reactionsSpecies[45][1] = Tau_P1_id;
	reactionsValues[45][1] = -1.0000000000;
	reactionsSpecies[45][2] = GSK3b_p53_P_id;
	reactionsValues[45][2] = 1.0000000000;
	reactionsSpecies[45][3] = Tau_P2_id;
	reactionsValues[45][3] = 1.0000000000;
	reactionsSpecies[46][0] = GSK3b_id;
	reactionsValues[46][0] = -1.0000000000;
	reactionsSpecies[46][1] = Tau_id;
	reactionsValues[46][1] = -1.0000000000;
	reactionsSpecies[46][2] = GSK3b_id;
	reactionsValues[46][2] = 1.0000000000;
	reactionsSpecies[46][3] = Tau_P1_id;
	reactionsValues[46][3] = 1.0000000000;
	reactionsSpecies[47][0] = GSK3b_id;
	reactionsValues[47][0] = -1.0000000000;
	reactionsSpecies[47][1] = Tau_P1_id;
	reactionsValues[47][1] = -1.0000000000;
	reactionsSpecies[47][2] = GSK3b_id;
	reactionsValues[47][2] = 1.0000000000;
	reactionsSpecies[47][3] = Tau_P2_id;
	reactionsValues[47][3] = 1.0000000000;
	reactionsSpecies[48][0] = Tau_P2_id;
	reactionsValues[48][0] = -1.0000000000;
	reactionsSpecies[48][1] = PP1_id;
	reactionsValues[48][1] = -1.0000000000;
	reactionsSpecies[48][2] = Tau_P1_id;
	reactionsValues[48][2] = 1.0000000000;
	reactionsSpecies[48][3] = PP1_id;
	reactionsValues[48][3] = 1.0000000000;
	reactionsSpecies[49][0] = Tau_P1_id;
	reactionsValues[49][0] = -1.0000000000;
	reactionsSpecies[49][1] = PP1_id;
	reactionsValues[49][1] = -1.0000000000;
	reactionsSpecies[49][2] = Tau_id;
	reactionsValues[49][2] = 1.0000000000;
	reactionsSpecies[49][3] = PP1_id;
	reactionsValues[49][3] = 1.0000000000;
	reactionsSpecies[50][0] = Tau_P1_id;
	reactionsValues[50][0] = -2.0000000000;
	reactionsSpecies[50][1] = AggTau_id;
	reactionsValues[50][1] = 2.0000000000;
	reactionsSpecies[51][0] = Tau_P1_id;
	reactionsValues[51][0] = -1.0000000000;
	reactionsSpecies[51][1] = AggTau_id;
	reactionsValues[51][1] = -1.0000000000;
	reactionsSpecies[51][2] = AggTau_id;
	reactionsValues[51][2] = 2.0000000000;
	reactionsSpecies[52][0] = Tau_P2_id;
	reactionsValues[52][0] = -2.0000000000;
	reactionsSpecies[52][1] = AggTau_id;
	reactionsValues[52][1] = 2.0000000000;
	reactionsSpecies[53][0] = Tau_P2_id;
	reactionsValues[53][0] = -1.0000000000;
	reactionsSpecies[53][1] = AggTau_id;
	reactionsValues[53][1] = -1.0000000000;
	reactionsSpecies[53][2] = AggTau_id;
	reactionsValues[53][2] = 2.0000000000;
	reactionsSpecies[54][0] = Tau_id;
	reactionsValues[54][0] = -2.0000000000;
	reactionsSpecies[54][1] = AggTau_id;
	reactionsValues[54][1] = 2.0000000000;
	reactionsSpecies[55][0] = Tau_id;
	reactionsValues[55][0] = -1.0000000000;
	reactionsSpecies[55][1] = AggTau_id;
	reactionsValues[55][1] = -1.0000000000;
	reactionsSpecies[55][2] = AggTau_id;
	reactionsValues[55][2] = 2.0000000000;
	reactionsSpecies[56][0] = AggTau_id;
	reactionsValues[56][0] = -2.0000000000;
	reactionsSpecies[56][1] = NFT_id;
	reactionsValues[56][1] = 2.0000000000;
	reactionsSpecies[57][0] = AggTau_id;
	reactionsValues[57][0] = -1.0000000000;
	reactionsSpecies[57][1] = NFT_id;
	reactionsValues[57][1] = -1.0000000000;
	reactionsSpecies[57][2] = NFT_id;
	reactionsValues[57][2] = 2.0000000000;
	reactionsSpecies[58][0] = AggTau_id;
	reactionsValues[58][0] = -1.0000000000;
	reactionsSpecies[58][1] = Proteasome_id;
	reactionsValues[58][1] = -1.0000000000;
	reactionsSpecies[58][2] = AggTau_Proteasome_id;
	reactionsValues[58][2] = 1.0000000000;
	reactionsSpecies[59][0] = Abeta_id;
	reactionsValues[59][0] = 1.0000000000;
	reactionsSpecies[60][0] = GSK3b_p53_id;
	reactionsValues[60][0] = -1.0000000000;
	reactionsSpecies[60][1] = Abeta_id;
	reactionsValues[60][1] = 1.0000000000;
	reactionsSpecies[60][2] = GSK3b_p53_id;
	reactionsValues[60][2] = 1.0000000000;
	reactionsSpecies[61][0] = GSK3b_p53_P_id;
	reactionsValues[61][0] = -1.0000000000;
	reactionsSpecies[61][1] = Abeta_id;
	reactionsValues[61][1] = 1.0000000000;
	reactionsSpecies[61][2] = GSK3b_p53_P_id;
	reactionsValues[61][2] = 1.0000000000;
	reactionsSpecies[62][0] = AbetaDimer_id;
	reactionsValues[62][0] = -1.0000000000;
	reactionsSpecies[62][1] = Proteasome_id;
	reactionsValues[62][1] = -1.0000000000;
	reactionsSpecies[62][2] = AggAbeta_Proteasome_id;
	reactionsValues[62][2] = 1.0000000000;
	reactionsSpecies[63][0] = Abeta_id;
	reactionsValues[63][0] = -1.0000000000;
	reactionsSpecies[64][0] = Abeta_id;
	reactionsValues[64][0] = -1.0000000000;
	reactionsSpecies[64][1] = p53_mRNA_id;
	reactionsValues[64][1] = 1.0000000000;
	reactionsSpecies[64][2] = Abeta_id;
	reactionsValues[64][2] = 1.0000000000;
	reactionsSpecies[65][0] = IR_id;
	reactionsValues[65][0] = -1.0000000000;
	reactionsSpecies[65][1] = IR_id;
	reactionsValues[65][1] = 1.0000000000;
	reactionsSpecies[65][2] = damDNA_id;
	reactionsValues[65][2] = 1.0000000000;
	reactionsSpecies[66][0] = damDNA_id;
	reactionsValues[66][0] = -1.0000000000;
	reactionsSpecies[67][0] = damDNA_id;
	reactionsValues[67][0] = -1.0000000000;
	reactionsSpecies[67][1] = ATMI_id;
	reactionsValues[67][1] = -1.0000000000;
	reactionsSpecies[67][2] = damDNA_id;
	reactionsValues[67][2] = 1.0000000000;
	reactionsSpecies[67][3] = ATMA_id;
	reactionsValues[67][3] = 1.0000000000;
	reactionsSpecies[68][0] = p53_id;
	reactionsValues[68][0] = -1.0000000000;
	reactionsSpecies[68][1] = ATMA_id;
	reactionsValues[68][1] = -1.0000000000;
	reactionsSpecies[68][2] = p53_P_id;
	reactionsValues[68][2] = 1.0000000000;
	reactionsSpecies[68][3] = ATMA_id;
	reactionsValues[68][3] = 1.0000000000;
	reactionsSpecies[69][0] = p53_P_id;
	reactionsValues[69][0] = -1.0000000000;
	reactionsSpecies[69][1] = p53_id;
	reactionsValues[69][1] = 1.0000000000;
	reactionsSpecies[70][0] = Mdm2_id;
	reactionsValues[70][0] = -1.0000000000;
	reactionsSpecies[70][1] = ATMA_id;
	reactionsValues[70][1] = -1.0000000000;
	reactionsSpecies[70][2] = Mdm2_P_id;
	reactionsValues[70][2] = 1.0000000000;
	reactionsSpecies[70][3] = ATMA_id;
	reactionsValues[70][3] = 1.0000000000;
	reactionsSpecies[71][0] = Mdm2_P_id;
	reactionsValues[71][0] = -1.0000000000;
	reactionsSpecies[71][1] = Mdm2_id;
	reactionsValues[71][1] = 1.0000000000;
	reactionsSpecies[72][0] = Mdm2_P_id;
	reactionsValues[72][0] = -1.0000000000;
	reactionsSpecies[72][1] = E2_Ub_id;
	reactionsValues[72][1] = -1.0000000000;
	reactionsSpecies[72][2] = Mdm2_P_Ub_id;
	reactionsValues[72][2] = 1.0000000000;
	reactionsSpecies[72][3] = E2_id;
	reactionsValues[72][3] = 1.0000000000;
	reactionsSpecies[73][0] = Mdm2_P_Ub_id;
	reactionsValues[73][0] = -1.0000000000;
	reactionsSpecies[73][1] = E2_Ub_id;
	reactionsValues[73][1] = -1.0000000000;
	reactionsSpecies[73][2] = Mdm2_P_Ub2_id;
	reactionsValues[73][2] = 1.0000000000;
	reactionsSpecies[73][3] = E2_id;
	reactionsValues[73][3] = 1.0000000000;
	reactionsSpecies[74][0] = Mdm2_P_Ub2_id;
	reactionsValues[74][0] = -1.0000000000;
	reactionsSpecies[74][1] = E2_Ub_id;
	reactionsValues[74][1] = -1.0000000000;
	reactionsSpecies[74][2] = Mdm2_P_Ub3_id;
	reactionsValues[74][2] = 1.0000000000;
	reactionsSpecies[74][3] = E2_id;
	reactionsValues[74][3] = 1.0000000000;
	reactionsSpecies[75][0] = Mdm2_P_Ub3_id;
	reactionsValues[75][0] = -1.0000000000;
	reactionsSpecies[75][1] = E2_Ub_id;
	reactionsValues[75][1] = -1.0000000000;
	reactionsSpecies[75][2] = Mdm2_P_Ub4_id;
	reactionsValues[75][2] = 1.0000000000;
	reactionsSpecies[75][3] = E2_id;
	reactionsValues[75][3] = 1.0000000000;
	reactionsSpecies[76][0] = Mdm2_P_Ub4_id;
	reactionsValues[76][0] = -1.0000000000;
	reactionsSpecies[76][1] = Mdm2DUB_id;
	reactionsValues[76][1] = -1.0000000000;
	reactionsSpecies[76][2] = Mdm2_P_Ub3_id;
	reactionsValues[76][2] = 1.0000000000;
	reactionsSpecies[76][3] = Mdm2DUB_id;
	reactionsValues[76][3] = 1.0000000000;
	reactionsSpecies[76][4] = Ub_id;
	reactionsValues[76][4] = 1.0000000000;
	reactionsSpecies[77][0] = Mdm2_P_Ub3_id;
	reactionsValues[77][0] = -1.0000000000;
	reactionsSpecies[77][1] = Mdm2DUB_id;
	reactionsValues[77][1] = -1.0000000000;
	reactionsSpecies[77][2] = Mdm2_P_Ub2_id;
	reactionsValues[77][2] = 1.0000000000;
	reactionsSpecies[77][3] = Mdm2DUB_id;
	reactionsValues[77][3] = 1.0000000000;
	reactionsSpecies[77][4] = Ub_id;
	reactionsValues[77][4] = 1.0000000000;
	reactionsSpecies[78][0] = Mdm2_P_Ub2_id;
	reactionsValues[78][0] = -1.0000000000;
	reactionsSpecies[78][1] = Mdm2DUB_id;
	reactionsValues[78][1] = -1.0000000000;
	reactionsSpecies[78][2] = Mdm2_P_Ub_id;
	reactionsValues[78][2] = 1.0000000000;
	reactionsSpecies[78][3] = Mdm2DUB_id;
	reactionsValues[78][3] = 1.0000000000;
	reactionsSpecies[78][4] = Ub_id;
	reactionsValues[78][4] = 1.0000000000;
	reactionsSpecies[79][0] = Mdm2_P_Ub_id;
	reactionsValues[79][0] = -1.0000000000;
	reactionsSpecies[79][1] = Mdm2DUB_id;
	reactionsValues[79][1] = -1.0000000000;
	reactionsSpecies[79][2] = Mdm2_P_id;
	reactionsValues[79][2] = 1.0000000000;
	reactionsSpecies[79][3] = Mdm2DUB_id;
	reactionsValues[79][3] = 1.0000000000;
	reactionsSpecies[79][4] = Ub_id;
	reactionsValues[79][4] = 1.0000000000;
	reactionsSpecies[80][0] = Mdm2_P_Ub4_id;
	reactionsValues[80][0] = -1.0000000000;
	reactionsSpecies[80][1] = Proteasome_id;
	reactionsValues[80][1] = -1.0000000000;
	reactionsSpecies[80][2] = Mdm2_P_Ub4_Proteasome_id;
	reactionsValues[80][2] = 1.0000000000;
	reactionsSpecies[81][0] = Mdm2_P_Ub4_Proteasome_id;
	reactionsValues[81][0] = -1.0000000000;
	reactionsSpecies[81][1] = Proteasome_id;
	reactionsValues[81][1] = 1.0000000000;
	reactionsSpecies[81][2] = Ub_id;
	reactionsValues[81][2] = 4.0000000000;
	reactionsSpecies[82][0] = ATMA_id;
	reactionsValues[82][0] = -1.0000000000;
	reactionsSpecies[82][1] = ATMI_id;
	reactionsValues[82][1] = 1.0000000000;
	reactionsSpecies[83][0] = Abeta_id;
	reactionsValues[83][0] = -1.0000000000;
	reactionsSpecies[83][1] = Abeta_id;
	reactionsValues[83][1] = 1.0000000000;
	reactionsSpecies[83][2] = ROS_id;
	reactionsValues[83][2] = 1.0000000000;
	reactionsSpecies[84][0] = AbetaPlaque_id;
	reactionsValues[84][0] = -1.0000000000;
	reactionsSpecies[84][1] = AbetaPlaque_id;
	reactionsValues[84][1] = 1.0000000000;
	reactionsSpecies[84][2] = ROS_id;
	reactionsValues[84][2] = 1.0000000000;
	reactionsSpecies[85][0] = AggAbeta_Proteasome_id;
	reactionsValues[85][0] = -1.0000000000;
	reactionsSpecies[85][1] = AggAbeta_Proteasome_id;
	reactionsValues[85][1] = 1.0000000000;
	reactionsSpecies[85][2] = ROS_id;
	reactionsValues[85][2] = 1.0000000000;
	reactionsSpecies[86][0] = ROS_id;
	reactionsValues[86][0] = -1.0000000000;
	reactionsSpecies[86][1] = ROS_id;
	reactionsValues[86][1] = 1.0000000000;
	reactionsSpecies[86][2] = damDNA_id;
	reactionsValues[86][2] = 1.0000000000;
	reactionsSpecies[87][0] = Tau_id;
	reactionsValues[87][0] = 1.0000000000;
	reactionsSpecies[88][0] = Tau_id;
	reactionsValues[88][0] = -1.0000000000;
	reactionsSpecies[88][1] = Proteasome_id;
	reactionsValues[88][1] = -1.0000000000;
	reactionsSpecies[88][2] = Proteasome_Tau_id;
	reactionsValues[88][2] = 1.0000000000;
	reactionsSpecies[89][0] = Proteasome_Tau_id;
	reactionsValues[89][0] = -1.0000000000;
	reactionsSpecies[89][1] = Proteasome_id;
	reactionsValues[89][1] = 1.0000000000;
	reactionsSpecies[90][0] = Abeta_id;
	reactionsValues[90][0] = -2.0000000000;
	reactionsSpecies[90][1] = AbetaDimer_id;
	reactionsValues[90][1] = 1.0000000000;
	reactionsSpecies[91][0] = AbetaDimer_id;
	reactionsValues[91][0] = -2.0000000000;
	reactionsSpecies[91][1] = AbetaPlaque_id;
	reactionsValues[91][1] = 1.0000000000;
	reactionsSpecies[92][0] = AbetaDimer_id;
	reactionsValues[92][0] = -1.0000000000;
	reactionsSpecies[92][1] = AbetaPlaque_id;
	reactionsValues[92][1] = -1.0000000000;
	reactionsSpecies[92][2] = AbetaPlaque_id;
	reactionsValues[92][2] = 2.0000000000;
	reactionsSpecies[93][0] = AbetaDimer_id;
	reactionsValues[93][0] = -1.0000000000;
	reactionsSpecies[93][1] = Abeta_id;
	reactionsValues[93][1] = 2.0000000000;
	reactionsSpecies[94][0] = AbetaPlaque_id;
	reactionsValues[94][0] = -1.0000000000;
	reactionsSpecies[94][1] = AbetaDimer_id;
	reactionsValues[94][1] = 1.0000000000;
	reactionsSpecies[94][2] = disaggPlaque1_id;
	reactionsValues[94][2] = 1.0000000000;
	reactionsSpecies[95][0] = AbetaPlaque_id;
	reactionsValues[95][0] = -1.0000000000;
	reactionsSpecies[95][1] = antiAb_id;
	reactionsValues[95][1] = -1.0000000000;
	reactionsSpecies[95][2] = AbetaDimer_id;
	reactionsValues[95][2] = 1.0000000000;
	reactionsSpecies[95][3] = antiAb_id;
	reactionsValues[95][3] = 1.0000000000;
	reactionsSpecies[95][4] = disaggPlaque2_id;
	reactionsValues[95][4] = 1.0000000000;
	reactionsSpecies[96][0] = Abeta_id;
	reactionsValues[96][0] = -1.0000000000;
	reactionsSpecies[96][1] = antiAb_id;
	reactionsValues[96][1] = -1.0000000000;
	reactionsSpecies[96][2] = Abeta_antiAb_id;
	reactionsValues[96][2] = 1.0000000000;
	reactionsSpecies[97][0] = AbetaDimer_id;
	reactionsValues[97][0] = -1.0000000000;
	reactionsSpecies[97][1] = antiAb_id;
	reactionsValues[97][1] = -1.0000000000;
	reactionsSpecies[97][2] = AbetaDimer_antiAb_id;
	reactionsValues[97][2] = 1.0000000000;
	reactionsSpecies[98][0] = Abeta_antiAb_id;
	reactionsValues[98][0] = -1.0000000000;
	reactionsSpecies[98][1] = antiAb_id;
	reactionsValues[98][1] = 1.0000000000;
	reactionsSpecies[99][0] = AbetaDimer_antiAb_id;
	reactionsValues[99][0] = -1.0000000000;
	reactionsSpecies[99][1] = antiAb_id;
	reactionsValues[99][1] = 1.0000000000;
	reactionsSpecies[100][0] = GliaI_id;
	reactionsValues[100][0] = -1.0000000000;
	reactionsSpecies[100][1] = AbetaPlaque_id;
	reactionsValues[100][1] = -1.0000000000;
	reactionsSpecies[100][2] = GliaM1_id;
	reactionsValues[100][2] = 1.0000000000;
	reactionsSpecies[100][3] = AbetaPlaque_id;
	reactionsValues[100][3] = 1.0000000000;
	reactionsSpecies[101][0] = GliaM1_id;
	reactionsValues[101][0] = -1.0000000000;
	reactionsSpecies[101][1] = AbetaPlaque_id;
	reactionsValues[101][1] = -1.0000000000;
	reactionsSpecies[101][2] = GliaM2_id;
	reactionsValues[101][2] = 1.0000000000;
	reactionsSpecies[101][3] = AbetaPlaque_id;
	reactionsValues[101][3] = 1.0000000000;
	reactionsSpecies[102][0] = GliaM2_id;
	reactionsValues[102][0] = -1.0000000000;
	reactionsSpecies[102][1] = antiAb_id;
	reactionsValues[102][1] = -1.0000000000;
	reactionsSpecies[102][2] = GliaA_id;
	reactionsValues[102][2] = 1.0000000000;
	reactionsSpecies[102][3] = antiAb_id;
	reactionsValues[102][3] = 1.0000000000;
	reactionsSpecies[103][0] = GliaA_id;
	reactionsValues[103][0] = -1.0000000000;
	reactionsSpecies[103][1] = GliaM2_id;
	reactionsValues[103][1] = 1.0000000000;
	reactionsSpecies[104][0] = GliaM2_id;
	reactionsValues[104][0] = -1.0000000000;
	reactionsSpecies[104][1] = GliaM1_id;
	reactionsValues[104][1] = 1.0000000000;
	reactionsSpecies[105][0] = GliaM1_id;
	reactionsValues[105][0] = -1.0000000000;
	reactionsSpecies[105][1] = GliaI_id;
	reactionsValues[105][1] = 1.0000000000;
	reactionsSpecies[106][0] = AbetaPlaque_id;
	reactionsValues[106][0] = -1.0000000000;
	reactionsSpecies[106][1] = GliaA_id;
	reactionsValues[106][1] = -1.0000000000;
	reactionsSpecies[106][2] = AbetaPlaque_GliaA_id;
	reactionsValues[106][2] = 1.0000000000;
	reactionsSpecies[107][0] = AbetaPlaque_GliaA_id;
	reactionsValues[107][0] = -1.0000000000;
	reactionsSpecies[107][1] = AbetaPlaque_id;
	reactionsValues[107][1] = 1.0000000000;
	reactionsSpecies[107][2] = GliaA_id;
	reactionsValues[107][2] = 1.0000000000;
	reactionsSpecies[108][0] = AbetaPlaque_GliaA_id;
	reactionsValues[108][0] = -1.0000000000;
	reactionsSpecies[108][1] = GliaA_id;
	reactionsValues[108][1] = 1.0000000000;
	reactionsSpecies[108][2] = degAbetaGlia_id;
	reactionsValues[108][2] = 1.0000000000;
	reactionsSpecies[109][0] = AbetaPlaque_GliaA_id;
	reactionsValues[109][0] = -1.0000000000;
	reactionsSpecies[109][1] = AbetaPlaque_GliaA_id;
	reactionsValues[109][1] = 1.0000000000;
	reactionsSpecies[109][2] = ROS_id;
	reactionsValues[109][2] = 1.0000000000;
	reactionsSpecies[110][0] = antiAb_id;
	reactionsValues[110][0] = -1.0000000000;
	reactionsSpecies[111][0] = ROS_id;
	reactionsValues[111][0] = -1.0000000000;
	curandState localState = state[threadIdx.x];
	while (time < endTime && time < (numberOfExecutions + 1)*segmentSize) {
		cummulative_p[0] = ksynp53mRNA * Source;
		cummulative_p[1] = cummulative_p[0] + kdegp53mRNA * p53_mRNA;
		cummulative_p[2] = cummulative_p[1] + ksynMdm2 * Mdm2_mRNA;
		cummulative_p[3] = cummulative_p[2] + ksynMdm2mRNA * p53;
		cummulative_p[4] = cummulative_p[3] + ksynMdm2mRNA * p53_P;
		cummulative_p[5] = cummulative_p[4] + ksynMdm2mRNAGSK3bp53 * GSK3b_p53;
		cummulative_p[6] = cummulative_p[5] + ksynMdm2mRNAGSK3bp53 * GSK3b_p53_P;
		cummulative_p[7] = cummulative_p[6] + kdegMdm2mRNA * Mdm2_mRNA;
		cummulative_p[8] = cummulative_p[7] + kbinMdm2p53 * p53 * Mdm2;
		cummulative_p[9] = cummulative_p[8] + krelMdm2p53 * Mdm2_p53;
		cummulative_p[10] = cummulative_p[9] + kbinGSK3bp53 * GSK3b * p53;
		cummulative_p[11] = cummulative_p[10] + krelGSK3bp53 * GSK3b_p53;
		cummulative_p[12] = cummulative_p[11] + kbinGSK3bp53 * GSK3b * p53_P;
		cummulative_p[13] = cummulative_p[12] + krelGSK3bp53 * GSK3b_p53_P;
		cummulative_p[14] = cummulative_p[13] + kbinE1Ub * E1 * Ub * ATP / (5000 + ATP);
		cummulative_p[15] = cummulative_p[14] + kbinE2Ub * E2 * E1_Ub;
		cummulative_p[16] = cummulative_p[15] + kMdm2Ub * Mdm2 * E2_Ub;
		cummulative_p[17] = cummulative_p[16] + kMdm2PolyUb * Mdm2_Ub * E2_Ub;
		cummulative_p[18] = cummulative_p[17] + kMdm2PolyUb * Mdm2_Ub2 * E2_Ub;
		cummulative_p[19] = cummulative_p[18] + kMdm2PolyUb * Mdm2_Ub3 * E2_Ub;
		cummulative_p[20] = cummulative_p[19] + kactDUBMdm2 * Mdm2_Ub4 * Mdm2DUB;
		cummulative_p[21] = cummulative_p[20] + kactDUBMdm2 * Mdm2_Ub3 * Mdm2DUB;
		cummulative_p[22] = cummulative_p[21] + kactDUBMdm2 * Mdm2_Ub2 * Mdm2DUB;
		cummulative_p[23] = cummulative_p[22] + kactDUBMdm2 * Mdm2_Ub * Mdm2DUB;
		cummulative_p[24] = cummulative_p[23] + kbinProt * Mdm2_Ub4 * Proteasome;
		cummulative_p[25] = cummulative_p[24] + kdegMdm2 * Mdm2_Ub4_Proteasome * kproteff;
		cummulative_p[26] = cummulative_p[25] + ksynp53 * p53_mRNA;
		cummulative_p[27] = cummulative_p[26] + kp53Ub * E2_Ub * Mdm2_p53;
		cummulative_p[28] = cummulative_p[27] + kp53PolyUb * Mdm2_p53_Ub * E2_Ub;
		cummulative_p[29] = cummulative_p[28] + kp53PolyUb * Mdm2_p53_Ub2 * E2_Ub;
		cummulative_p[30] = cummulative_p[29] + kp53PolyUb * Mdm2_p53_Ub3 * E2_Ub;
		cummulative_p[31] = cummulative_p[30] + kactDUBp53 * Mdm2_p53_Ub4 * p53DUB;
		cummulative_p[32] = cummulative_p[31] + kactDUBp53 * Mdm2_p53_Ub3 * p53DUB;
		cummulative_p[33] = cummulative_p[32] + kactDUBp53 * Mdm2_p53_Ub2 * p53DUB;
		cummulative_p[34] = cummulative_p[33] + kactDUBp53 * Mdm2_p53_Ub * p53DUB;
		cummulative_p[35] = cummulative_p[34] + kphosMdm2GSK3b * Mdm2_p53_Ub4 * GSK3b;
		cummulative_p[36] = cummulative_p[35] + kphosMdm2GSK3bp53 * Mdm2_p53_Ub4 * GSK3b_p53;
		cummulative_p[37] = cummulative_p[36] + kphosMdm2GSK3bp53 * Mdm2_p53_Ub4 * GSK3b_p53_P;
		cummulative_p[38] = cummulative_p[37] + kbinProt * Mdm2_P1_p53_Ub4 * Proteasome;
		cummulative_p[39] = cummulative_p[38] + kdegp53 * kproteff * p53_Ub4_Proteasome * ATP / (5000 + ATP);
		cummulative_p[40] = cummulative_p[39] + kbinMTTau * Tau;
		cummulative_p[41] = cummulative_p[40] + krelMTTau * MT_Tau;
		cummulative_p[42] = cummulative_p[41] + kphospTauGSK3bp53 * GSK3b_p53 * Tau;
		cummulative_p[43] = cummulative_p[42] + kphospTauGSK3bp53 * GSK3b_p53 * Tau_P1;
		cummulative_p[44] = cummulative_p[43] + kphospTauGSK3bp53 * GSK3b_p53_P * Tau;
		cummulative_p[45] = cummulative_p[44] + kphospTauGSK3bp53 * GSK3b_p53_P * Tau_P1;
		cummulative_p[46] = cummulative_p[45] + kphospTauGSK3b * GSK3b * Tau;
		cummulative_p[47] = cummulative_p[46] + kphospTauGSK3b * GSK3b * Tau_P1;
		cummulative_p[48] = cummulative_p[47] + kdephospTau * Tau_P2 * PP1;
		cummulative_p[49] = cummulative_p[48] + kdephospTau * Tau_P1 * PP1;
		cummulative_p[50] = cummulative_p[49] + kaggTauP1 * Tau_P1 * (Tau_P1 - 1) * 0.5;
		cummulative_p[51] = cummulative_p[50] + kaggTauP1 * Tau_P1 * AggTau;
		cummulative_p[52] = cummulative_p[51] + kaggTauP2 * Tau_P2 * (Tau_P2 - 1) * 0.5;
		cummulative_p[53] = cummulative_p[52] + kaggTauP2 * Tau_P2 * AggTau;
		cummulative_p[54] = cummulative_p[53] + kaggTau * Tau * (Tau - 1) * 0.5;
		cummulative_p[55] = cummulative_p[54] + kaggTau * Tau * AggTau;
		cummulative_p[56] = cummulative_p[55] + ktangfor * AggTau * (AggTau - 1) * 0.5;
		cummulative_p[57] = cummulative_p[56] + ktangfor * AggTau * NFT;
		cummulative_p[58] = cummulative_p[57] + kinhibprot * AggTau * Proteasome;
		cummulative_p[59] = cummulative_p[58] + kprodAbeta * Source;
		cummulative_p[60] = cummulative_p[59] + kprodAbeta2 * GSK3b_p53;
		cummulative_p[61] = cummulative_p[60] + kprodAbeta2 * GSK3b_p53_P;
		cummulative_p[62] = cummulative_p[61] + kinhibprot * AbetaDimer * Proteasome;
		cummulative_p[63] = cummulative_p[62] + kdegAbeta * Abeta;
		cummulative_p[64] = cummulative_p[63] + ksynp53mRNAAbeta * Abeta;
		cummulative_p[65] = cummulative_p[64] + kdam * IR;
		cummulative_p[66] = cummulative_p[65] + krepair * damDNA;
		cummulative_p[67] = cummulative_p[66] + kactATM * damDNA * ATMI;
		cummulative_p[68] = cummulative_p[67] + kphosp53 * p53 * ATMA;
		cummulative_p[69] = cummulative_p[68] + kdephosp53 * p53_P;
		cummulative_p[70] = cummulative_p[69] + kphosMdm2 * Mdm2 * ATMA;
		cummulative_p[71] = cummulative_p[70] + kdephosMdm2 * Mdm2_P;
		cummulative_p[72] = cummulative_p[71] + kMdm2PUb * Mdm2_P * E2_Ub;
		cummulative_p[73] = cummulative_p[72] + kMdm2PolyUb * Mdm2_P_Ub * E2_Ub;
		cummulative_p[74] = cummulative_p[73] + kMdm2PolyUb * Mdm2_P_Ub2 * E2_Ub;
		cummulative_p[75] = cummulative_p[74] + kMdm2PolyUb * Mdm2_P_Ub3 * E2_Ub;
		cummulative_p[76] = cummulative_p[75] + kactDUBMdm2 * Mdm2_P_Ub4 * Mdm2DUB;
		cummulative_p[77] = cummulative_p[76] + kactDUBMdm2 * Mdm2_P_Ub3 * Mdm2DUB;
		cummulative_p[78] = cummulative_p[77] + kactDUBMdm2 * Mdm2_P_Ub2 * Mdm2DUB;
		cummulative_p[79] = cummulative_p[78] + kactDUBMdm2 * Mdm2_P_Ub * Mdm2DUB;
		cummulative_p[80] = cummulative_p[79] + kbinProt * Mdm2_P_Ub4 * Proteasome;
		cummulative_p[81] = cummulative_p[80] + kdegMdm2 * Mdm2_P_Ub4_Proteasome * kproteff;
		cummulative_p[82] = cummulative_p[81] + kinactATM * ATMA;
		cummulative_p[83] = cummulative_p[82] + kgenROSAbeta * Abeta;
		cummulative_p[84] = cummulative_p[83] + kgenROSPlaque * AbetaPlaque;
		cummulative_p[85] = cummulative_p[84] + kgenROSAbeta * AggAbeta_Proteasome;
		cummulative_p[86] = cummulative_p[85] + kdamROS * ROS;
		cummulative_p[87] = cummulative_p[86] + ksynTau * Source;
		cummulative_p[88] = cummulative_p[87] + kbinTauProt * Tau * Proteasome;
		cummulative_p[89] = cummulative_p[88] + kdegTau20SProt * Proteasome_Tau;
		cummulative_p[90] = cummulative_p[89] + kaggAbeta * Abeta * (Abeta - 1) * 0.5;
		cummulative_p[91] = cummulative_p[90] + kpf * AbetaDimer * (AbetaDimer - 1) * 0.5;
		cummulative_p[92] = cummulative_p[91] + kpg * AbetaDimer * pow(AbetaPlaque, 2) / (pow(kpghalf, 2) + pow(AbetaPlaque, 2));
		cummulative_p[93] = cummulative_p[92] + kdisaggAbeta * AbetaDimer;
		cummulative_p[94] = cummulative_p[93] + kdisaggAbeta1 * AbetaPlaque;
		cummulative_p[95] = cummulative_p[94] + kdisaggAbeta2 * antiAb * AbetaPlaque;
		cummulative_p[96] = cummulative_p[95] + kbinAbantiAb * Abeta * antiAb;
		cummulative_p[97] = cummulative_p[96] + kbinAbantiAb * AbetaDimer * antiAb;
		cummulative_p[98] = cummulative_p[97] + 10 * kdegAbeta * Abeta_antiAb;
		cummulative_p[99] = cummulative_p[98] + 10 * kdegAbeta * AbetaDimer_antiAb;
		cummulative_p[100] = cummulative_p[99] + kactglia1 * GliaI * AbetaPlaque;
		cummulative_p[101] = cummulative_p[100] + kactglia1 * GliaM1 * AbetaPlaque;
		cummulative_p[102] = cummulative_p[101] + kactglia2 * GliaM2 * antiAb;
		cummulative_p[103] = cummulative_p[102] + kinactglia1 * GliaA;
		cummulative_p[104] = cummulative_p[103] + kinactglia2 * GliaM2;
		cummulative_p[105] = cummulative_p[104] + kinactglia2 * GliaM1;
		cummulative_p[106] = cummulative_p[105] + kbinAbetaGlia * AbetaPlaque * GliaA;
		cummulative_p[107] = cummulative_p[106] + krelAbetaGlia * AbetaPlaque_GliaA;
		cummulative_p[108] = cummulative_p[107] + kdegAbetaGlia * AbetaPlaque_GliaA;
		cummulative_p[109] = cummulative_p[108] + kgenROSGlia * AbetaPlaque_GliaA;
		cummulative_p[110] = cummulative_p[109] + kdegAntiAb * antiAb;
		cummulative_p[111] = cummulative_p[110] + kremROS * ROS;
		if (time >= segmentSize * numberOfExecutions + step * stepCount) {
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 0], species[0]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 1], species[1]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 2], species[2]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 3], species[3]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 4], species[4]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 5], species[5]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 6], species[6]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 7], species[7]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 8], species[8]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 9], species[9]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 10], species[10]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 11], species[11]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 12], species[12]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 13], species[13]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 14], species[14]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 15], species[15]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 16], species[16]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 17], species[17]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 18], species[18]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 19], species[19]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 20], species[20]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 21], species[21]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 22], species[22]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 23], species[23]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 24], species[24]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 25], species[25]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 26], species[26]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 27], species[27]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 28], species[28]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 29], species[29]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 30], species[30]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 31], species[31]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 32], species[32]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 33], species[33]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 34], species[34]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 35], species[35]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 36], species[36]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 37], species[37]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 38], species[38]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 39], species[39]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 40], species[40]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 41], species[41]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 42], species[42]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 43], species[43]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 44], species[44]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 45], species[45]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 46], species[46]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 47], species[47]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 48], species[48]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 49], species[49]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 50], species[50]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 51], species[51]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 52], species[52]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 53], species[53]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 54], species[54]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 55], species[55]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 56], species[56]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 57], species[57]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 58], species[58]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 59], species[59]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 60], species[60]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 61], species[61]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 62], species[62]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 63], species[63]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 64], species[64]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 65], species[65]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 66], species[66]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 67], species[67]);
			atomicAdd(&output[69 * 34 * numberOfExecutions + stepCount * 69 + 68], species[68]);
			stepCount++;
		}
		sum_p = cummulative_p[111];
		random = curand_uniform(&localState);
		if (sum_p > 0) timeStep = -log(random) / sum_p;
		else break;
		random = curand_uniform(&localState);
		random *= sum_p;
		indexMin = 0;
		indexMax = 111;
		while (indexMax > indexMin) {
			reaction = (indexMin + indexMax) / 2;
			if (cummulative_p[reaction - 1] <= random) {
				if (cummulative_p[reaction] > random) {
					break;
				}
				else {
					indexMin = reaction;
				}
			}
			else {
				indexMax = reaction;
			}
		}
		for (int i = 0; i < 5; i++) {
			if (reactionsSpecies[reaction][i] == -1) { break; }
			species[reactionsSpecies[reaction][i]] += reactionsValues[reaction][i];
		}
		if (triggerEvent0 == 0 && time >= 345600) {
			triggerEvent0 = 1;
			antiAb += 50;
		}
		time += timeStep;
	}
	Mdm2_global[threadIdx.x] = species[0];
	p53_global[threadIdx.x] = species[1];
	Mdm2_p53_global[threadIdx.x] = species[2];
	Mdm2_mRNA_global[threadIdx.x] = species[3];
	p53_mRNA_global[threadIdx.x] = species[4];
	ATMA_global[threadIdx.x] = species[5];
	ATMI_global[threadIdx.x] = species[6];
	p53_P_global[threadIdx.x] = species[7];
	Mdm2_P_global[threadIdx.x] = species[8];
	IR_global[threadIdx.x] = species[9];
	ROS_global[threadIdx.x] = species[10];
	damDNA_global[threadIdx.x] = species[11];
	E1_global[threadIdx.x] = species[12];
	E2_global[threadIdx.x] = species[13];
	E1_Ub_global[threadIdx.x] = species[14];
	E2_Ub_global[threadIdx.x] = species[15];
	Proteasome_global[threadIdx.x] = species[16];
	Ub_global[threadIdx.x] = species[17];
	p53DUB_global[threadIdx.x] = species[18];
	Mdm2DUB_global[threadIdx.x] = species[19];
	DUB_global[threadIdx.x] = species[20];
	Mdm2_p53_Ub_global[threadIdx.x] = species[21];
	Mdm2_p53_Ub2_global[threadIdx.x] = species[22];
	Mdm2_p53_Ub3_global[threadIdx.x] = species[23];
	Mdm2_p53_Ub4_global[threadIdx.x] = species[24];
	Mdm2_P1_p53_Ub4_global[threadIdx.x] = species[25];
	Mdm2_Ub_global[threadIdx.x] = species[26];
	Mdm2_Ub2_global[threadIdx.x] = species[27];
	Mdm2_Ub3_global[threadIdx.x] = species[28];
	Mdm2_Ub4_global[threadIdx.x] = species[29];
	Mdm2_P_Ub_global[threadIdx.x] = species[30];
	Mdm2_P_Ub2_global[threadIdx.x] = species[31];
	Mdm2_P_Ub3_global[threadIdx.x] = species[32];
	Mdm2_P_Ub4_global[threadIdx.x] = species[33];
	p53_Ub4_Proteasome_global[threadIdx.x] = species[34];
	Mdm2_Ub4_Proteasome_global[threadIdx.x] = species[35];
	Mdm2_P_Ub4_Proteasome_global[threadIdx.x] = species[36];
	GSK3b_global[threadIdx.x] = species[37];
	GSK3b_p53_global[threadIdx.x] = species[38];
	GSK3b_p53_P_global[threadIdx.x] = species[39];
	Abeta_global[threadIdx.x] = species[40];
	AggAbeta_Proteasome_global[threadIdx.x] = species[41];
	AbetaPlaque_global[threadIdx.x] = species[42];
	Tau_global[threadIdx.x] = species[43];
	Tau_P1_global[threadIdx.x] = species[44];
	Tau_P2_global[threadIdx.x] = species[45];
	MT_Tau_global[threadIdx.x] = species[46];
	AggTau_global[threadIdx.x] = species[47];
	AggTau_Proteasome_global[threadIdx.x] = species[48];
	Proteasome_Tau_global[threadIdx.x] = species[49];
	PP1_global[threadIdx.x] = species[50];
	NFT_global[threadIdx.x] = species[51];
	ATP_global[threadIdx.x] = species[52];
	ADP_global[threadIdx.x] = species[53];
	AMP_global[threadIdx.x] = species[54];
	AbetaDimer_global[threadIdx.x] = species[55];
	AbetaPlaque_GliaA_global[threadIdx.x] = species[56];
	GliaI_global[threadIdx.x] = species[57];
	GliaM1_global[threadIdx.x] = species[58];
	GliaM2_global[threadIdx.x] = species[59];
	GliaA_global[threadIdx.x] = species[60];
	antiAb_global[threadIdx.x] = species[61];
	Abeta_antiAb_global[threadIdx.x] = species[62];
	AbetaDimer_antiAb_global[threadIdx.x] = species[63];
	degAbetaGlia_global[threadIdx.x] = species[64];
	disaggPlaque1_global[threadIdx.x] = species[65];
	disaggPlaque2_global[threadIdx.x] = species[66];
	Source_global[threadIdx.x] = species[67];
	Sink_global[threadIdx.x] = species[68];
	state[threadIdx.x] = localState;
}

__global__
void initCurand(curandState* state, unsigned long long seed) {
	curand_init(seed, threadIdx.x, 0, &state[threadIdx.x]);
}

int main()
{
	cudaError_t cudaStatus;
	float* output;
	float* dev_output;
	output = (float*)malloc(167 * 69 * sizeof(float));
	for (int i = 0; i < 167 * 69; i++) {
		output[i] = 0;
	}
	cudaStatus = cudaMalloc(&dev_output, 167 * 69 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_output, output, 167 * 69 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float host_Mdm2 = 5.0000000000;
	float* dev_Mdm2;
	cudaStatus = cudaMalloc(&dev_Mdm2, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2, &host_Mdm2, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2_global;
	cudaStatus = cudaMalloc(&Mdm2_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_p53 = 5.0000000000;
	float* dev_p53;
	cudaStatus = cudaMalloc(&dev_p53, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_p53, &host_p53, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* p53_global;
	cudaStatus = cudaMalloc(&p53_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Mdm2_p53 = 95.0000000000;
	float* dev_Mdm2_p53;
	cudaStatus = cudaMalloc(&dev_Mdm2_p53, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2_p53, &host_Mdm2_p53, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2_p53_global;
	cudaStatus = cudaMalloc(&Mdm2_p53_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Mdm2_mRNA = 10.0000000000;
	float* dev_Mdm2_mRNA;
	cudaStatus = cudaMalloc(&dev_Mdm2_mRNA, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2_mRNA, &host_Mdm2_mRNA, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2_mRNA_global;
	cudaStatus = cudaMalloc(&Mdm2_mRNA_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_p53_mRNA = 10.0000000000;
	float* dev_p53_mRNA;
	cudaStatus = cudaMalloc(&dev_p53_mRNA, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_p53_mRNA, &host_p53_mRNA, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* p53_mRNA_global;
	cudaStatus = cudaMalloc(&p53_mRNA_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_ATMA = 0.0000000000;
	float* dev_ATMA;
	cudaStatus = cudaMalloc(&dev_ATMA, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_ATMA, &host_ATMA, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* ATMA_global;
	cudaStatus = cudaMalloc(&ATMA_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_ATMI = 200.0000000000;
	float* dev_ATMI;
	cudaStatus = cudaMalloc(&dev_ATMI, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_ATMI, &host_ATMI, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* ATMI_global;
	cudaStatus = cudaMalloc(&ATMI_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_p53_P = 0.0000000000;
	float* dev_p53_P;
	cudaStatus = cudaMalloc(&dev_p53_P, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_p53_P, &host_p53_P, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* p53_P_global;
	cudaStatus = cudaMalloc(&p53_P_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Mdm2_P = 0.0000000000;
	float* dev_Mdm2_P;
	cudaStatus = cudaMalloc(&dev_Mdm2_P, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2_P, &host_Mdm2_P, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2_P_global;
	cudaStatus = cudaMalloc(&Mdm2_P_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_IR = 0.0000000000;
	float* dev_IR;
	cudaStatus = cudaMalloc(&dev_IR, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_IR, &host_IR, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* IR_global;
	cudaStatus = cudaMalloc(&IR_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_ROS = 0.0000000000;
	float* dev_ROS;
	cudaStatus = cudaMalloc(&dev_ROS, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_ROS, &host_ROS, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* ROS_global;
	cudaStatus = cudaMalloc(&ROS_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_damDNA = 0.0000000000;
	float* dev_damDNA;
	cudaStatus = cudaMalloc(&dev_damDNA, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_damDNA, &host_damDNA, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* damDNA_global;
	cudaStatus = cudaMalloc(&damDNA_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_E1 = 100.0000000000;
	float* dev_E1;
	cudaStatus = cudaMalloc(&dev_E1, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_E1, &host_E1, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* E1_global;
	cudaStatus = cudaMalloc(&E1_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_E2 = 100.0000000000;
	float* dev_E2;
	cudaStatus = cudaMalloc(&dev_E2, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_E2, &host_E2, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* E2_global;
	cudaStatus = cudaMalloc(&E2_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_E1_Ub = 0.0000000000;
	float* dev_E1_Ub;
	cudaStatus = cudaMalloc(&dev_E1_Ub, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_E1_Ub, &host_E1_Ub, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* E1_Ub_global;
	cudaStatus = cudaMalloc(&E1_Ub_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_E2_Ub = 0.0000000000;
	float* dev_E2_Ub;
	cudaStatus = cudaMalloc(&dev_E2_Ub, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_E2_Ub, &host_E2_Ub, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* E2_Ub_global;
	cudaStatus = cudaMalloc(&E2_Ub_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Proteasome = 500.0000000000;
	float* dev_Proteasome;
	cudaStatus = cudaMalloc(&dev_Proteasome, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Proteasome, &host_Proteasome, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Proteasome_global;
	cudaStatus = cudaMalloc(&Proteasome_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Ub = 4000.0000000000;
	float* dev_Ub;
	cudaStatus = cudaMalloc(&dev_Ub, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Ub, &host_Ub, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Ub_global;
	cudaStatus = cudaMalloc(&Ub_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_p53DUB = 200.0000000000;
	float* dev_p53DUB;
	cudaStatus = cudaMalloc(&dev_p53DUB, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_p53DUB, &host_p53DUB, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* p53DUB_global;
	cudaStatus = cudaMalloc(&p53DUB_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Mdm2DUB = 200.0000000000;
	float* dev_Mdm2DUB;
	cudaStatus = cudaMalloc(&dev_Mdm2DUB, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2DUB, &host_Mdm2DUB, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2DUB_global;
	cudaStatus = cudaMalloc(&Mdm2DUB_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_DUB = 200.0000000000;
	float* dev_DUB;
	cudaStatus = cudaMalloc(&dev_DUB, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_DUB, &host_DUB, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* DUB_global;
	cudaStatus = cudaMalloc(&DUB_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Mdm2_p53_Ub = 0.0000000000;
	float* dev_Mdm2_p53_Ub;
	cudaStatus = cudaMalloc(&dev_Mdm2_p53_Ub, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2_p53_Ub, &host_Mdm2_p53_Ub, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2_p53_Ub_global;
	cudaStatus = cudaMalloc(&Mdm2_p53_Ub_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Mdm2_p53_Ub2 = 0.0000000000;
	float* dev_Mdm2_p53_Ub2;
	cudaStatus = cudaMalloc(&dev_Mdm2_p53_Ub2, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2_p53_Ub2, &host_Mdm2_p53_Ub2, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2_p53_Ub2_global;
	cudaStatus = cudaMalloc(&Mdm2_p53_Ub2_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Mdm2_p53_Ub3 = 0.0000000000;
	float* dev_Mdm2_p53_Ub3;
	cudaStatus = cudaMalloc(&dev_Mdm2_p53_Ub3, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2_p53_Ub3, &host_Mdm2_p53_Ub3, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2_p53_Ub3_global;
	cudaStatus = cudaMalloc(&Mdm2_p53_Ub3_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Mdm2_p53_Ub4 = 0.0000000000;
	float* dev_Mdm2_p53_Ub4;
	cudaStatus = cudaMalloc(&dev_Mdm2_p53_Ub4, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2_p53_Ub4, &host_Mdm2_p53_Ub4, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2_p53_Ub4_global;
	cudaStatus = cudaMalloc(&Mdm2_p53_Ub4_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Mdm2_P1_p53_Ub4 = 0.0000000000;
	float* dev_Mdm2_P1_p53_Ub4;
	cudaStatus = cudaMalloc(&dev_Mdm2_P1_p53_Ub4, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2_P1_p53_Ub4, &host_Mdm2_P1_p53_Ub4, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2_P1_p53_Ub4_global;
	cudaStatus = cudaMalloc(&Mdm2_P1_p53_Ub4_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Mdm2_Ub = 0.0000000000;
	float* dev_Mdm2_Ub;
	cudaStatus = cudaMalloc(&dev_Mdm2_Ub, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2_Ub, &host_Mdm2_Ub, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2_Ub_global;
	cudaStatus = cudaMalloc(&Mdm2_Ub_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Mdm2_Ub2 = 0.0000000000;
	float* dev_Mdm2_Ub2;
	cudaStatus = cudaMalloc(&dev_Mdm2_Ub2, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2_Ub2, &host_Mdm2_Ub2, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2_Ub2_global;
	cudaStatus = cudaMalloc(&Mdm2_Ub2_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Mdm2_Ub3 = 0.0000000000;
	float* dev_Mdm2_Ub3;
	cudaStatus = cudaMalloc(&dev_Mdm2_Ub3, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2_Ub3, &host_Mdm2_Ub3, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2_Ub3_global;
	cudaStatus = cudaMalloc(&Mdm2_Ub3_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Mdm2_Ub4 = 0.0000000000;
	float* dev_Mdm2_Ub4;
	cudaStatus = cudaMalloc(&dev_Mdm2_Ub4, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2_Ub4, &host_Mdm2_Ub4, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2_Ub4_global;
	cudaStatus = cudaMalloc(&Mdm2_Ub4_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Mdm2_P_Ub = 0.0000000000;
	float* dev_Mdm2_P_Ub;
	cudaStatus = cudaMalloc(&dev_Mdm2_P_Ub, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2_P_Ub, &host_Mdm2_P_Ub, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2_P_Ub_global;
	cudaStatus = cudaMalloc(&Mdm2_P_Ub_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Mdm2_P_Ub2 = 0.0000000000;
	float* dev_Mdm2_P_Ub2;
	cudaStatus = cudaMalloc(&dev_Mdm2_P_Ub2, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2_P_Ub2, &host_Mdm2_P_Ub2, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2_P_Ub2_global;
	cudaStatus = cudaMalloc(&Mdm2_P_Ub2_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Mdm2_P_Ub3 = 0.0000000000;
	float* dev_Mdm2_P_Ub3;
	cudaStatus = cudaMalloc(&dev_Mdm2_P_Ub3, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2_P_Ub3, &host_Mdm2_P_Ub3, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2_P_Ub3_global;
	cudaStatus = cudaMalloc(&Mdm2_P_Ub3_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Mdm2_P_Ub4 = 0.0000000000;
	float* dev_Mdm2_P_Ub4;
	cudaStatus = cudaMalloc(&dev_Mdm2_P_Ub4, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2_P_Ub4, &host_Mdm2_P_Ub4, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2_P_Ub4_global;
	cudaStatus = cudaMalloc(&Mdm2_P_Ub4_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_p53_Ub4_Proteasome = 0.0000000000;
	float* dev_p53_Ub4_Proteasome;
	cudaStatus = cudaMalloc(&dev_p53_Ub4_Proteasome, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_p53_Ub4_Proteasome, &host_p53_Ub4_Proteasome, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* p53_Ub4_Proteasome_global;
	cudaStatus = cudaMalloc(&p53_Ub4_Proteasome_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Mdm2_Ub4_Proteasome = 0.0000000000;
	float* dev_Mdm2_Ub4_Proteasome;
	cudaStatus = cudaMalloc(&dev_Mdm2_Ub4_Proteasome, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2_Ub4_Proteasome, &host_Mdm2_Ub4_Proteasome, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2_Ub4_Proteasome_global;
	cudaStatus = cudaMalloc(&Mdm2_Ub4_Proteasome_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Mdm2_P_Ub4_Proteasome = 0.0000000000;
	float* dev_Mdm2_P_Ub4_Proteasome;
	cudaStatus = cudaMalloc(&dev_Mdm2_P_Ub4_Proteasome, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Mdm2_P_Ub4_Proteasome, &host_Mdm2_P_Ub4_Proteasome, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Mdm2_P_Ub4_Proteasome_global;
	cudaStatus = cudaMalloc(&Mdm2_P_Ub4_Proteasome_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_GSK3b = 500.0000000000;
	float* dev_GSK3b;
	cudaStatus = cudaMalloc(&dev_GSK3b, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_GSK3b, &host_GSK3b, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* GSK3b_global;
	cudaStatus = cudaMalloc(&GSK3b_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_GSK3b_p53 = 0.0000000000;
	float* dev_GSK3b_p53;
	cudaStatus = cudaMalloc(&dev_GSK3b_p53, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_GSK3b_p53, &host_GSK3b_p53, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* GSK3b_p53_global;
	cudaStatus = cudaMalloc(&GSK3b_p53_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_GSK3b_p53_P = 0.0000000000;
	float* dev_GSK3b_p53_P;
	cudaStatus = cudaMalloc(&dev_GSK3b_p53_P, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_GSK3b_p53_P, &host_GSK3b_p53_P, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* GSK3b_p53_P_global;
	cudaStatus = cudaMalloc(&GSK3b_p53_P_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Abeta = 0.0000000000;
	float* dev_Abeta;
	cudaStatus = cudaMalloc(&dev_Abeta, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Abeta, &host_Abeta, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Abeta_global;
	cudaStatus = cudaMalloc(&Abeta_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_AggAbeta_Proteasome = 0.0000000000;
	float* dev_AggAbeta_Proteasome;
	cudaStatus = cudaMalloc(&dev_AggAbeta_Proteasome, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_AggAbeta_Proteasome, &host_AggAbeta_Proteasome, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* AggAbeta_Proteasome_global;
	cudaStatus = cudaMalloc(&AggAbeta_Proteasome_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_AbetaPlaque = 0.0000000000;
	float* dev_AbetaPlaque;
	cudaStatus = cudaMalloc(&dev_AbetaPlaque, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_AbetaPlaque, &host_AbetaPlaque, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* AbetaPlaque_global;
	cudaStatus = cudaMalloc(&AbetaPlaque_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Tau = 0.0000000000;
	float* dev_Tau;
	cudaStatus = cudaMalloc(&dev_Tau, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Tau, &host_Tau, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Tau_global;
	cudaStatus = cudaMalloc(&Tau_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Tau_P1 = 0.0000000000;
	float* dev_Tau_P1;
	cudaStatus = cudaMalloc(&dev_Tau_P1, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Tau_P1, &host_Tau_P1, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Tau_P1_global;
	cudaStatus = cudaMalloc(&Tau_P1_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Tau_P2 = 0.0000000000;
	float* dev_Tau_P2;
	cudaStatus = cudaMalloc(&dev_Tau_P2, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Tau_P2, &host_Tau_P2, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Tau_P2_global;
	cudaStatus = cudaMalloc(&Tau_P2_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_MT_Tau = 100.0000000000;
	float* dev_MT_Tau;
	cudaStatus = cudaMalloc(&dev_MT_Tau, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_MT_Tau, &host_MT_Tau, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* MT_Tau_global;
	cudaStatus = cudaMalloc(&MT_Tau_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_AggTau = 0.0000000000;
	float* dev_AggTau;
	cudaStatus = cudaMalloc(&dev_AggTau, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_AggTau, &host_AggTau, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* AggTau_global;
	cudaStatus = cudaMalloc(&AggTau_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_AggTau_Proteasome = 0.0000000000;
	float* dev_AggTau_Proteasome;
	cudaStatus = cudaMalloc(&dev_AggTau_Proteasome, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_AggTau_Proteasome, &host_AggTau_Proteasome, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* AggTau_Proteasome_global;
	cudaStatus = cudaMalloc(&AggTau_Proteasome_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Proteasome_Tau = 0.0000000000;
	float* dev_Proteasome_Tau;
	cudaStatus = cudaMalloc(&dev_Proteasome_Tau, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Proteasome_Tau, &host_Proteasome_Tau, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Proteasome_Tau_global;
	cudaStatus = cudaMalloc(&Proteasome_Tau_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_PP1 = 50.0000000000;
	float* dev_PP1;
	cudaStatus = cudaMalloc(&dev_PP1, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_PP1, &host_PP1, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* PP1_global;
	cudaStatus = cudaMalloc(&PP1_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_NFT = 0.0000000000;
	float* dev_NFT;
	cudaStatus = cudaMalloc(&dev_NFT, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_NFT, &host_NFT, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* NFT_global;
	cudaStatus = cudaMalloc(&NFT_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_ATP = 10000.0000000000;
	float* dev_ATP;
	cudaStatus = cudaMalloc(&dev_ATP, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_ATP, &host_ATP, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* ATP_global;
	cudaStatus = cudaMalloc(&ATP_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_ADP = 1000.0000000000;
	float* dev_ADP;
	cudaStatus = cudaMalloc(&dev_ADP, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_ADP, &host_ADP, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* ADP_global;
	cudaStatus = cudaMalloc(&ADP_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_AMP = 1000.0000000000;
	float* dev_AMP;
	cudaStatus = cudaMalloc(&dev_AMP, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_AMP, &host_AMP, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* AMP_global;
	cudaStatus = cudaMalloc(&AMP_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_AbetaDimer = 0.0000000000;
	float* dev_AbetaDimer;
	cudaStatus = cudaMalloc(&dev_AbetaDimer, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_AbetaDimer, &host_AbetaDimer, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* AbetaDimer_global;
	cudaStatus = cudaMalloc(&AbetaDimer_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_AbetaPlaque_GliaA = 0.0000000000;
	float* dev_AbetaPlaque_GliaA;
	cudaStatus = cudaMalloc(&dev_AbetaPlaque_GliaA, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_AbetaPlaque_GliaA, &host_AbetaPlaque_GliaA, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* AbetaPlaque_GliaA_global;
	cudaStatus = cudaMalloc(&AbetaPlaque_GliaA_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_GliaI = 100.0000000000;
	float* dev_GliaI;
	cudaStatus = cudaMalloc(&dev_GliaI, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_GliaI, &host_GliaI, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* GliaI_global;
	cudaStatus = cudaMalloc(&GliaI_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_GliaM1 = 0.0000000000;
	float* dev_GliaM1;
	cudaStatus = cudaMalloc(&dev_GliaM1, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_GliaM1, &host_GliaM1, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* GliaM1_global;
	cudaStatus = cudaMalloc(&GliaM1_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_GliaM2 = 0.0000000000;
	float* dev_GliaM2;
	cudaStatus = cudaMalloc(&dev_GliaM2, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_GliaM2, &host_GliaM2, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* GliaM2_global;
	cudaStatus = cudaMalloc(&GliaM2_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_GliaA = 0.0000000000;
	float* dev_GliaA;
	cudaStatus = cudaMalloc(&dev_GliaA, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_GliaA, &host_GliaA, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* GliaA_global;
	cudaStatus = cudaMalloc(&GliaA_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_antiAb = 0.0000000000;
	float* dev_antiAb;
	cudaStatus = cudaMalloc(&dev_antiAb, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_antiAb, &host_antiAb, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* antiAb_global;
	cudaStatus = cudaMalloc(&antiAb_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Abeta_antiAb = 0.0000000000;
	float* dev_Abeta_antiAb;
	cudaStatus = cudaMalloc(&dev_Abeta_antiAb, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Abeta_antiAb, &host_Abeta_antiAb, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Abeta_antiAb_global;
	cudaStatus = cudaMalloc(&Abeta_antiAb_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_AbetaDimer_antiAb = 0.0000000000;
	float* dev_AbetaDimer_antiAb;
	cudaStatus = cudaMalloc(&dev_AbetaDimer_antiAb, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_AbetaDimer_antiAb, &host_AbetaDimer_antiAb, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* AbetaDimer_antiAb_global;
	cudaStatus = cudaMalloc(&AbetaDimer_antiAb_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_degAbetaGlia = 0.0000000000;
	float* dev_degAbetaGlia;
	cudaStatus = cudaMalloc(&dev_degAbetaGlia, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_degAbetaGlia, &host_degAbetaGlia, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* degAbetaGlia_global;
	cudaStatus = cudaMalloc(&degAbetaGlia_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_disaggPlaque1 = 0.0000000000;
	float* dev_disaggPlaque1;
	cudaStatus = cudaMalloc(&dev_disaggPlaque1, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_disaggPlaque1, &host_disaggPlaque1, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* disaggPlaque1_global;
	cudaStatus = cudaMalloc(&disaggPlaque1_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_disaggPlaque2 = 0.0000000000;
	float* dev_disaggPlaque2;
	cudaStatus = cudaMalloc(&dev_disaggPlaque2, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_disaggPlaque2, &host_disaggPlaque2, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* disaggPlaque2_global;
	cudaStatus = cudaMalloc(&disaggPlaque2_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Source = 1.0000000000;
	float* dev_Source;
	cudaStatus = cudaMalloc(&dev_Source, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Source, &host_Source, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Source_global;
	cudaStatus = cudaMalloc(&Source_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float host_Sink = 1.0000000000;
	float* dev_Sink;
	cudaStatus = cudaMalloc(&dev_Sink, sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_Sink, &host_Sink, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* Sink_global;
	cudaStatus = cudaMalloc(&Sink_global, 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	curandState *devStates;
	CUDA_CALL(cudaMalloc((void **)&devStates, 32 * sizeof(curandState)));
	initCurand << <1, 32 >> > (devStates, SEED);
	for (int i = 0; i < 5; i++) {
		simulate << <1, 32 >> > (i, dev_output, devStates, 60.0000000000, 10000.0000000000, 2000, dev_Mdm2, Mdm2_global, dev_p53, p53_global, dev_Mdm2_p53, Mdm2_p53_global, dev_Mdm2_mRNA, Mdm2_mRNA_global, dev_p53_mRNA, p53_mRNA_global, dev_ATMA, ATMA_global, dev_ATMI, ATMI_global, dev_p53_P, p53_P_global, dev_Mdm2_P, Mdm2_P_global, dev_IR, IR_global, dev_ROS, ROS_global, dev_damDNA, damDNA_global, dev_E1, E1_global, dev_E2, E2_global, dev_E1_Ub, E1_Ub_global, dev_E2_Ub, E2_Ub_global, dev_Proteasome, Proteasome_global, dev_Ub, Ub_global, dev_p53DUB, p53DUB_global, dev_Mdm2DUB, Mdm2DUB_global, dev_DUB, DUB_global, dev_Mdm2_p53_Ub, Mdm2_p53_Ub_global, dev_Mdm2_p53_Ub2, Mdm2_p53_Ub2_global, dev_Mdm2_p53_Ub3, Mdm2_p53_Ub3_global, dev_Mdm2_p53_Ub4, Mdm2_p53_Ub4_global, dev_Mdm2_P1_p53_Ub4, Mdm2_P1_p53_Ub4_global, dev_Mdm2_Ub, Mdm2_Ub_global, dev_Mdm2_Ub2, Mdm2_Ub2_global, dev_Mdm2_Ub3, Mdm2_Ub3_global, dev_Mdm2_Ub4, Mdm2_Ub4_global, dev_Mdm2_P_Ub, Mdm2_P_Ub_global, dev_Mdm2_P_Ub2, Mdm2_P_Ub2_global, dev_Mdm2_P_Ub3, Mdm2_P_Ub3_global, dev_Mdm2_P_Ub4, Mdm2_P_Ub4_global, dev_p53_Ub4_Proteasome, p53_Ub4_Proteasome_global, dev_Mdm2_Ub4_Proteasome, Mdm2_Ub4_Proteasome_global, dev_Mdm2_P_Ub4_Proteasome, Mdm2_P_Ub4_Proteasome_global, dev_GSK3b, GSK3b_global, dev_GSK3b_p53, GSK3b_p53_global, dev_GSK3b_p53_P, GSK3b_p53_P_global, dev_Abeta, Abeta_global, dev_AggAbeta_Proteasome, AggAbeta_Proteasome_global, dev_AbetaPlaque, AbetaPlaque_global, dev_Tau, Tau_global, dev_Tau_P1, Tau_P1_global, dev_Tau_P2, Tau_P2_global, dev_MT_Tau, MT_Tau_global, dev_AggTau, AggTau_global, dev_AggTau_Proteasome, AggTau_Proteasome_global, dev_Proteasome_Tau, Proteasome_Tau_global, dev_PP1, PP1_global, dev_NFT, NFT_global, dev_ATP, ATP_global, dev_ADP, ADP_global, dev_AMP, AMP_global, dev_AbetaDimer, AbetaDimer_global, dev_AbetaPlaque_GliaA, AbetaPlaque_GliaA_global, dev_GliaI, GliaI_global, dev_GliaM1, GliaM1_global, dev_GliaM2, GliaM2_global, dev_GliaA, GliaA_global, dev_antiAb, antiAb_global, dev_Abeta_antiAb, Abeta_antiAb_global, dev_AbetaDimer_antiAb, AbetaDimer_antiAb_global, dev_degAbetaGlia, degAbetaGlia_global, dev_disaggPlaque1, disaggPlaque1_global, dev_disaggPlaque2, disaggPlaque2_global, dev_Source, Source_global, dev_Sink, Sink_global);

		cudaStatus = cudaGetLastError(); if (cudaStatus != cudaSuccess) { fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }

		cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }
	}


	cudaStatus = cudaMemcpy(output, dev_output, 167 * 69 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2, dev_Mdm2, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_p53, dev_p53, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2_p53, dev_Mdm2_p53, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2_mRNA, dev_Mdm2_mRNA, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_p53_mRNA, dev_p53_mRNA, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_ATMA, dev_ATMA, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_ATMI, dev_ATMI, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_p53_P, dev_p53_P, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2_P, dev_Mdm2_P, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_IR, dev_IR, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_ROS, dev_ROS, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_damDNA, dev_damDNA, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_E1, dev_E1, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_E2, dev_E2, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_E1_Ub, dev_E1_Ub, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_E2_Ub, dev_E2_Ub, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Proteasome, dev_Proteasome, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Ub, dev_Ub, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_p53DUB, dev_p53DUB, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2DUB, dev_Mdm2DUB, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_DUB, dev_DUB, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2_p53_Ub, dev_Mdm2_p53_Ub, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2_p53_Ub2, dev_Mdm2_p53_Ub2, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2_p53_Ub3, dev_Mdm2_p53_Ub3, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2_p53_Ub4, dev_Mdm2_p53_Ub4, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2_P1_p53_Ub4, dev_Mdm2_P1_p53_Ub4, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2_Ub, dev_Mdm2_Ub, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2_Ub2, dev_Mdm2_Ub2, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2_Ub3, dev_Mdm2_Ub3, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2_Ub4, dev_Mdm2_Ub4, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2_P_Ub, dev_Mdm2_P_Ub, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2_P_Ub2, dev_Mdm2_P_Ub2, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2_P_Ub3, dev_Mdm2_P_Ub3, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2_P_Ub4, dev_Mdm2_P_Ub4, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_p53_Ub4_Proteasome, dev_p53_Ub4_Proteasome, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2_Ub4_Proteasome, dev_Mdm2_Ub4_Proteasome, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Mdm2_P_Ub4_Proteasome, dev_Mdm2_P_Ub4_Proteasome, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_GSK3b, dev_GSK3b, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_GSK3b_p53, dev_GSK3b_p53, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_GSK3b_p53_P, dev_GSK3b_p53_P, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Abeta, dev_Abeta, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_AggAbeta_Proteasome, dev_AggAbeta_Proteasome, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_AbetaPlaque, dev_AbetaPlaque, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Tau, dev_Tau, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Tau_P1, dev_Tau_P1, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Tau_P2, dev_Tau_P2, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_MT_Tau, dev_MT_Tau, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_AggTau, dev_AggTau, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_AggTau_Proteasome, dev_AggTau_Proteasome, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Proteasome_Tau, dev_Proteasome_Tau, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_PP1, dev_PP1, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_NFT, dev_NFT, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_ATP, dev_ATP, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_ADP, dev_ADP, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_AMP, dev_AMP, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_AbetaDimer, dev_AbetaDimer, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_AbetaPlaque_GliaA, dev_AbetaPlaque_GliaA, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_GliaI, dev_GliaI, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_GliaM1, dev_GliaM1, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_GliaM2, dev_GliaM2, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_GliaA, dev_GliaA, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_antiAb, dev_antiAb, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Abeta_antiAb, dev_Abeta_antiAb, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_AbetaDimer_antiAb, dev_AbetaDimer_antiAb, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_degAbetaGlia, dev_degAbetaGlia, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_disaggPlaque1, dev_disaggPlaque1, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_disaggPlaque2, dev_disaggPlaque2, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Source, dev_Source, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(&host_Sink, dev_Sink, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	FILE* results = fopen("results.csv", "w");
	if (results == NULL) {
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
	for (int i = 0; i < 167; i++) {
		fprintf(results, "%.10lf", 60.0000000000*i);
		for (int j = 0; j < 69; j++) {
			fprintf(results, ", %.10lf", output[69 * i + j] / 32);
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
