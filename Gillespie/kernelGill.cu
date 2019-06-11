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
void simulate(int numberOfExecutions, float* output, curandState *state, float step, float endTime, float segmentSize, float* species_global) {
	int reaction, stepCount = 0;
	int indexMin, indexMax;
	float time = numberOfExecutions * segmentSize;
	float sum_p, timeStep, random;
	float cummulative_p[112];
	int triggerEvent0 = 0;
	if (time >= 345600) { triggerEvent0 = 1; }
	float species[69];
	if (numberOfExecutions == 0) {
		species[0] = species_global[0];
	}
	else {
		species[0] = species_global[Mdm2_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[1] = species_global[1];
	}
	else {
		species[1] = species_global[p53_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[2] = species_global[2];
	}
	else {
		species[2] = species_global[Mdm2_p53_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[3] = species_global[3];
	}
	else {
		species[3] = species_global[Mdm2_mRNA_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[4] = species_global[4];
	}
	else {
		species[4] = species_global[p53_mRNA_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[5] = species_global[5];
	}
	else {
		species[5] = species_global[ATMA_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[6] = species_global[6];
	}
	else {
		species[6] = species_global[ATMI_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[7] = species_global[7];
	}
	else {
		species[7] = species_global[p53_P_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[8] = species_global[8];
	}
	else {
		species[8] = species_global[Mdm2_P_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[9] = species_global[9];
	}
	else {
		species[9] = species_global[IR_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[10] = species_global[10];
	}
	else {
		species[10] = species_global[ROS_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[11] = species_global[11];
	}
	else {
		species[11] = species_global[damDNA_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[12] = species_global[12];
	}
	else {
		species[12] = species_global[E1_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[13] = species_global[13];
	}
	else {
		species[13] = species_global[E2_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[14] = species_global[14];
	}
	else {
		species[14] = species_global[E1_Ub_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[15] = species_global[15];
	}
	else {
		species[15] = species_global[E2_Ub_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[16] = species_global[16];
	}
	else {
		species[16] = species_global[Proteasome_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[17] = species_global[17];
	}
	else {
		species[17] = species_global[Ub_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[18] = species_global[18];
	}
	else {
		species[18] = species_global[p53DUB_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[19] = species_global[19];
	}
	else {
		species[19] = species_global[Mdm2DUB_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[20] = species_global[20];
	}
	else {
		species[20] = species_global[DUB_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[21] = species_global[21];
	}
	else {
		species[21] = species_global[Mdm2_p53_Ub_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[22] = species_global[22];
	}
	else {
		species[22] = species_global[Mdm2_p53_Ub2_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[23] = species_global[23];
	}
	else {
		species[23] = species_global[Mdm2_p53_Ub3_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[24] = species_global[24];
	}
	else {
		species[24] = species_global[Mdm2_p53_Ub4_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[25] = species_global[25];
	}
	else {
		species[25] = species_global[Mdm2_P1_p53_Ub4_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[26] = species_global[26];
	}
	else {
		species[26] = species_global[Mdm2_Ub_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[27] = species_global[27];
	}
	else {
		species[27] = species_global[Mdm2_Ub2_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[28] = species_global[28];
	}
	else {
		species[28] = species_global[Mdm2_Ub3_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[29] = species_global[29];
	}
	else {
		species[29] = species_global[Mdm2_Ub4_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[30] = species_global[30];
	}
	else {
		species[30] = species_global[Mdm2_P_Ub_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[31] = species_global[31];
	}
	else {
		species[31] = species_global[Mdm2_P_Ub2_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[32] = species_global[32];
	}
	else {
		species[32] = species_global[Mdm2_P_Ub3_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[33] = species_global[33];
	}
	else {
		species[33] = species_global[Mdm2_P_Ub4_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[34] = species_global[34];
	}
	else {
		species[34] = species_global[p53_Ub4_Proteasome_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[35] = species_global[35];
	}
	else {
		species[35] = species_global[Mdm2_Ub4_Proteasome_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[36] = species_global[36];
	}
	else {
		species[36] = species_global[Mdm2_P_Ub4_Proteasome_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[37] = species_global[37];
	}
	else {
		species[37] = species_global[GSK3b_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[38] = species_global[38];
	}
	else {
		species[38] = species_global[GSK3b_p53_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[39] = species_global[39];
	}
	else {
		species[39] = species_global[GSK3b_p53_P_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[40] = species_global[40];
	}
	else {
		species[40] = species_global[Abeta_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[41] = species_global[41];
	}
	else {
		species[41] = species_global[AggAbeta_Proteasome_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[42] = species_global[42];
	}
	else {
		species[42] = species_global[AbetaPlaque_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[43] = species_global[43];
	}
	else {
		species[43] = species_global[Tau_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[44] = species_global[44];
	}
	else {
		species[44] = species_global[Tau_P1_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[45] = species_global[45];
	}
	else {
		species[45] = species_global[Tau_P2_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[46] = species_global[46];
	}
	else {
		species[46] = species_global[MT_Tau_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[47] = species_global[47];
	}
	else {
		species[47] = species_global[AggTau_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[48] = species_global[48];
	}
	else {
		species[48] = species_global[AggTau_Proteasome_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[49] = species_global[49];
	}
	else {
		species[49] = species_global[Proteasome_Tau_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[50] = species_global[50];
	}
	else {
		species[50] = species_global[PP1_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[51] = species_global[51];
	}
	else {
		species[51] = species_global[NFT_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[52] = species_global[52];
	}
	else {
		species[52] = species_global[ATP_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[53] = species_global[53];
	}
	else {
		species[53] = species_global[ADP_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[54] = species_global[54];
	}
	else {
		species[54] = species_global[AMP_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[55] = species_global[55];
	}
	else {
		species[55] = species_global[AbetaDimer_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[56] = species_global[56];
	}
	else {
		species[56] = species_global[AbetaPlaque_GliaA_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[57] = species_global[57];
	}
	else {
		species[57] = species_global[GliaI_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[58] = species_global[58];
	}
	else {
		species[58] = species_global[GliaM1_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[59] = species_global[59];
	}
	else {
		species[59] = species_global[GliaM2_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[60] = species_global[60];
	}
	else {
		species[60] = species_global[GliaA_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[61] = species_global[61];
	}
	else {
		species[61] = species_global[antiAb_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[62] = species_global[62];
	}
	else {
		species[62] = species_global[Abeta_antiAb_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[63] = species_global[63];
	}
	else {
		species[63] = species_global[AbetaDimer_antiAb_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[64] = species_global[64];
	}
	else {
		species[64] = species_global[degAbetaGlia_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[65] = species_global[65];
	}
	else {
		species[65] = species_global[disaggPlaque1_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[66] = species_global[66];
	}
	else {
		species[66] = species_global[disaggPlaque2_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[67] = species_global[67];
	}
	else {
		species[67] = species_global[Source_id * 32 + threadIdx.x];
	}
	if (numberOfExecutions == 0) {
		species[68] = species_global[68];
	}
	else {
		species[68] = species_global[Sink_id * 32 + threadIdx.x];
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
			if (cummulative_p[reaction] <= random) {
				if (cummulative_p[reaction + 1] > random) {
					reaction++;
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
	species_global[Mdm2_id * 32 + threadIdx.x] = species[0];
	species_global[p53_id * 32 + threadIdx.x] = species[1];
	species_global[Mdm2_p53_id * 32 + threadIdx.x] = species[2];
	species_global[Mdm2_mRNA_id * 32 + threadIdx.x] = species[3];
	species_global[p53_mRNA_id * 32 + threadIdx.x] = species[4];
	species_global[ATMA_id * 32 + threadIdx.x] = species[5];
	species_global[ATMI_id * 32 + threadIdx.x] = species[6];
	species_global[p53_P_id * 32 + threadIdx.x] = species[7];
	species_global[Mdm2_P_id * 32 + threadIdx.x] = species[8];
	species_global[IR_id * 32 + threadIdx.x] = species[9];
	species_global[ROS_id * 32 + threadIdx.x] = species[10];
	species_global[damDNA_id * 32 + threadIdx.x] = species[11];
	species_global[E1_id * 32 + threadIdx.x] = species[12];
	species_global[E2_id * 32 + threadIdx.x] = species[13];
	species_global[E1_Ub_id * 32 + threadIdx.x] = species[14];
	species_global[E2_Ub_id * 32 + threadIdx.x] = species[15];
	species_global[Proteasome_id * 32 + threadIdx.x] = species[16];
	species_global[Ub_id * 32 + threadIdx.x] = species[17];
	species_global[p53DUB_id * 32 + threadIdx.x] = species[18];
	species_global[Mdm2DUB_id * 32 + threadIdx.x] = species[19];
	species_global[DUB_id * 32 + threadIdx.x] = species[20];
	species_global[Mdm2_p53_Ub_id * 32 + threadIdx.x] = species[21];
	species_global[Mdm2_p53_Ub2_id * 32 + threadIdx.x] = species[22];
	species_global[Mdm2_p53_Ub3_id * 32 + threadIdx.x] = species[23];
	species_global[Mdm2_p53_Ub4_id * 32 + threadIdx.x] = species[24];
	species_global[Mdm2_P1_p53_Ub4_id * 32 + threadIdx.x] = species[25];
	species_global[Mdm2_Ub_id * 32 + threadIdx.x] = species[26];
	species_global[Mdm2_Ub2_id * 32 + threadIdx.x] = species[27];
	species_global[Mdm2_Ub3_id * 32 + threadIdx.x] = species[28];
	species_global[Mdm2_Ub4_id * 32 + threadIdx.x] = species[29];
	species_global[Mdm2_P_Ub_id * 32 + threadIdx.x] = species[30];
	species_global[Mdm2_P_Ub2_id * 32 + threadIdx.x] = species[31];
	species_global[Mdm2_P_Ub3_id * 32 + threadIdx.x] = species[32];
	species_global[Mdm2_P_Ub4_id * 32 + threadIdx.x] = species[33];
	species_global[p53_Ub4_Proteasome_id * 32 + threadIdx.x] = species[34];
	species_global[Mdm2_Ub4_Proteasome_id * 32 + threadIdx.x] = species[35];
	species_global[Mdm2_P_Ub4_Proteasome_id * 32 + threadIdx.x] = species[36];
	species_global[GSK3b_id * 32 + threadIdx.x] = species[37];
	species_global[GSK3b_p53_id * 32 + threadIdx.x] = species[38];
	species_global[GSK3b_p53_P_id * 32 + threadIdx.x] = species[39];
	species_global[Abeta_id * 32 + threadIdx.x] = species[40];
	species_global[AggAbeta_Proteasome_id * 32 + threadIdx.x] = species[41];
	species_global[AbetaPlaque_id * 32 + threadIdx.x] = species[42];
	species_global[Tau_id * 32 + threadIdx.x] = species[43];
	species_global[Tau_P1_id * 32 + threadIdx.x] = species[44];
	species_global[Tau_P2_id * 32 + threadIdx.x] = species[45];
	species_global[MT_Tau_id * 32 + threadIdx.x] = species[46];
	species_global[AggTau_id * 32 + threadIdx.x] = species[47];
	species_global[AggTau_Proteasome_id * 32 + threadIdx.x] = species[48];
	species_global[Proteasome_Tau_id * 32 + threadIdx.x] = species[49];
	species_global[PP1_id * 32 + threadIdx.x] = species[50];
	species_global[NFT_id * 32 + threadIdx.x] = species[51];
	species_global[ATP_id * 32 + threadIdx.x] = species[52];
	species_global[ADP_id * 32 + threadIdx.x] = species[53];
	species_global[AMP_id * 32 + threadIdx.x] = species[54];
	species_global[AbetaDimer_id * 32 + threadIdx.x] = species[55];
	species_global[AbetaPlaque_GliaA_id * 32 + threadIdx.x] = species[56];
	species_global[GliaI_id * 32 + threadIdx.x] = species[57];
	species_global[GliaM1_id * 32 + threadIdx.x] = species[58];
	species_global[GliaM2_id * 32 + threadIdx.x] = species[59];
	species_global[GliaA_id * 32 + threadIdx.x] = species[60];
	species_global[antiAb_id * 32 + threadIdx.x] = species[61];
	species_global[Abeta_antiAb_id * 32 + threadIdx.x] = species[62];
	species_global[AbetaDimer_antiAb_id * 32 + threadIdx.x] = species[63];
	species_global[degAbetaGlia_id * 32 + threadIdx.x] = species[64];
	species_global[disaggPlaque1_id * 32 + threadIdx.x] = species[65];
	species_global[disaggPlaque2_id * 32 + threadIdx.x] = species[66];
	species_global[Source_id * 32 + threadIdx.x] = species[67];
	species_global[Sink_id * 32 + threadIdx.x] = species[68];
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
	output = (float*)malloc(334 * 69 * sizeof(float));
	for (int i = 0; i < 334 * 69; i++) {
		output[i] = 0;
	}
	cudaStatus = cudaMalloc(&dev_output, 334 * 69 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_output, output, 334 * 69 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	float* species_global;
	cudaStatus = cudaMalloc(&species_global, 69 * 32 * sizeof(float));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	float init_species[69];
	init_species[0] = 5.0000000000;
	init_species[1] = 5.0000000000;
	init_species[2] = 95.0000000000;
	init_species[3] = 10.0000000000;
	init_species[4] = 10.0000000000;
	init_species[5] = 0.0000000000;
	init_species[6] = 200.0000000000;
	init_species[7] = 0.0000000000;
	init_species[8] = 0.0000000000;
	init_species[9] = 0.0000000000;
	init_species[10] = 0.0000000000;
	init_species[11] = 0.0000000000;
	init_species[12] = 100.0000000000;
	init_species[13] = 100.0000000000;
	init_species[14] = 0.0000000000;
	init_species[15] = 0.0000000000;
	init_species[16] = 500.0000000000;
	init_species[17] = 4000.0000000000;
	init_species[18] = 200.0000000000;
	init_species[19] = 200.0000000000;
	init_species[20] = 200.0000000000;
	init_species[21] = 0.0000000000;
	init_species[22] = 0.0000000000;
	init_species[23] = 0.0000000000;
	init_species[24] = 0.0000000000;
	init_species[25] = 0.0000000000;
	init_species[26] = 0.0000000000;
	init_species[27] = 0.0000000000;
	init_species[28] = 0.0000000000;
	init_species[29] = 0.0000000000;
	init_species[30] = 0.0000000000;
	init_species[31] = 0.0000000000;
	init_species[32] = 0.0000000000;
	init_species[33] = 0.0000000000;
	init_species[34] = 0.0000000000;
	init_species[35] = 0.0000000000;
	init_species[36] = 0.0000000000;
	init_species[37] = 500.0000000000;
	init_species[38] = 0.0000000000;
	init_species[39] = 0.0000000000;
	init_species[40] = 0.0000000000;
	init_species[41] = 0.0000000000;
	init_species[42] = 0.0000000000;
	init_species[43] = 0.0000000000;
	init_species[44] = 0.0000000000;
	init_species[45] = 0.0000000000;
	init_species[46] = 100.0000000000;
	init_species[47] = 0.0000000000;
	init_species[48] = 0.0000000000;
	init_species[49] = 0.0000000000;
	init_species[50] = 50.0000000000;
	init_species[51] = 0.0000000000;
	init_species[52] = 10000.0000000000;
	init_species[53] = 1000.0000000000;
	init_species[54] = 1000.0000000000;
	init_species[55] = 0.0000000000;
	init_species[56] = 0.0000000000;
	init_species[57] = 100.0000000000;
	init_species[58] = 0.0000000000;
	init_species[59] = 0.0000000000;
	init_species[60] = 0.0000000000;
	init_species[61] = 0.0000000000;
	init_species[62] = 0.0000000000;
	init_species[63] = 0.0000000000;
	init_species[64] = 0.0000000000;
	init_species[65] = 0.0000000000;
	init_species[66] = 0.0000000000;
	init_species[67] = 1.0000000000;
	init_species[68] = 1.0000000000;
	cudaStatus = cudaMemcpy(species_global, &init_species, sizeof(float) * 69, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	curandState *devStates;
	CUDA_CALL(cudaMalloc((void **)&devStates, 32 * sizeof(curandState)));
	initCurand << <1, 32 >> > (devStates, SEED);
	cudaEvent_t start, stop;
	float milliseconds;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	printf("GO\n");
	for (int i = 0; i < 10; i++) {
		cudaEventRecord(start);

		simulate << <1, 32 >> > (i, dev_output, devStates, 60.0000000000, 20000.0000000000, 2000, species_global); cudaStatus = cudaGetLastError(); if (cudaStatus != cudaSuccess) { fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }

		cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }
		cudaEventRecord(stop);
		printf("%d\n", i);
		cudaEventSynchronize(stop);
		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("TIME: %lf\n", milliseconds);
	}


	cudaStatus = cudaMemcpy(output, dev_output, 334 * 69 * sizeof(float), cudaMemcpyDeviceToHost);
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
	for (int i = 0; i < 334; i++) {
		fprintf(results, "%.10lf", 60.0000000000*i);
		for (int j = 0; j < 69; j++) {
			fprintf(results, ", %.10lf", output[69 * i + j] / 32);
		}
		fprintf(results, "\n");
	}
	fprintf(results, "\n");
Error:
	cudaFree(dev_output);
	cudaFree(species_global);

	return 0;
}
