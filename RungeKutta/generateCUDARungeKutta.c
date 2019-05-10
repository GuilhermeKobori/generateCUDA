#include <stdio.h>
#include <stdlib.h>

#include <sbml/SBMLTypes.h>

#include <math.h>
#include <stddef.h>
#include <string.h>

#define SQR(x) ((x)*(x))
#define SQRT(x) pow((x),(.5))

void generateCUDA(Model_t* m, double step, int simulations) {
	//get compartments and initial concentrations in key value pairs
	ListOf_t* species = Model_getListOfSpecies(m);
	ListOf_t* compartments = Model_getListOfCompartments(m);
	ListOf_t* parameters = Model_getListOfParameters(m);
	char* Key;
	double Value;
	FILE* defineReactionRates = fopen("RKdefineReactionRates", "w");	//contains metaprogramming for the reactionRates formulas
	FILE* defineConstants = fopen("RKdefineConstants", "w");			//contains constants for the simulation, containers, local parameters and global paramenter
	FILE* initializeSpecies = fopen("RKinitializeSpecies", "w");		//declares a variable and associates the value for each species, also creates and initializes devices variables
	FILE* kernelFunction = fopen("RKkernelFunction", "w");			//device function, loops around a simple Runge-Kutta simulation
	FILE* kernelCall = fopen("RKkernelCall", "w");					//declares the call for the kernel function
	FILE* kernelVariablesInit = fopen("RKkernelVariablesInit", "w");	//initializes the variables for the kernel function
	FILE* receiveData = fopen("RKreceiveData", "w");					//contains code that receives data from device to host
	FILE* freeDevice = fopen("RKfreeDevice", "w");					//contains free declarations for the device variables
	FILE* printResults = fopen("RKprintResults", "w");				//prints each species with the values received from device

	//error handling
	if (defineReactionRates == NULL)
	{
		printf("Error accessing defineReactionRates!");
		exit(1);
	}
	if (defineConstants == NULL)
	{
		printf("Error accessing defineConstants!");
		exit(1);
	}
	if (initializeSpecies == NULL)
	{
		printf("Error accessing initializeSpecies!");
		exit(1);
	}
	if (kernelFunction == NULL)
	{
		printf("Error accessing kernelFunction!");
		exit(1);
	}
	if (kernelCall == NULL)
	{
		printf("Error accessing kernelCall!");
		exit(1);
	}
	if (receiveData == NULL)
	{
		printf("Error accessing receiveData!");
		exit(1);
	}
	if (freeDevice == NULL)
	{
		printf("Error accessing freeDevice!");
		exit(1);
	}
	if (printResults == NULL)
	{
		printf("Error accessing printResults!");
		exit(1);
	}

	//get list of reactions and formulas
	KineticLaw_t *kl;

	fprintf(defineReactionRates, "#define simulateStepReaction(i) \\\n");
	fprintf(defineReactionRates, "switch (i) { \\\n");

	for (int i = 0; i < Model_getNumReactions(m); i++)
	{
		//save formulas processing time vs memory size
		if (Reaction_isSetKineticLaw(Model_getReaction(m, i)))
		{
			kl = Reaction_getKineticLaw(Model_getReaction(m, i));
			if (KineticLaw_isSetMath(kl))
			{
				fprintf(defineReactionRates, "case %d: \\\n", i);
				fprintf(defineReactionRates, "reactionRate = %s; \\\n", SBML_formulaToString(KineticLaw_getMath(kl)));

				//reactants
				ListOf_t* reactants = Reaction_getListOfReactants(Model_getReaction(m, i));
				for (int k = 0; k < Reaction_getNumReactants(Model_getReaction(m, i)); k++) {
					SpeciesReference_t* reactant = ListOf_get(reactants, k);
					if (Species_getConstant(ListOfSpecies_getById(species, SpeciesReference_getSpecies(reactant))) == 0) {
						fprintf(defineReactionRates, "atomicAdd(%s_aux, - step * %lf * reactionRate); \\\n", SpeciesReference_getSpecies(reactant), SpeciesReference_getStoichiometry(reactant));
					}
				}

				//products
				ListOf_t* products = Reaction_getListOfProducts(Model_getReaction(m, i));
				for (int k = 0; k < Reaction_getNumProducts(Model_getReaction(m, i)); k++) {
					SpeciesReference_t* product = ListOf_get(products, k);
					if (Species_getConstant(ListOfSpecies_getById(species, SpeciesReference_getSpecies(product))) == 0) {
						fprintf(defineReactionRates, "atomicAdd(%s_aux, step * %lf * reactionRate); \\\n", SpeciesReference_getSpecies(product), SpeciesReference_getStoichiometry(product));
					}
				}

				fprintf(defineReactionRates, "break; \\\n");
				for (int j = 0; j < KineticLaw_getNumParameters(kl); j++) {
					Parameter_t* p = KineticLaw_getParameter(kl, j);
					fprintf(defineConstants, "#define %s %lf\n", Parameter_getId(p), Parameter_getValue(p));
				}
			}
		}
	}


	fprintf(defineReactionRates, "} \\\n\n\n");

	//get initial values and constants

	fprintf(kernelFunction, "__global__\n");
	fprintf(kernelFunction, "void simulate (float step, int numSimulations");

	fprintf(initializeSpecies, "cudaError_t cudaStatus;\n");

	fprintf(kernelCall, "simulate<<<1, %d>>>(%lf, %d", Model_getNumReactions(m), step, simulations);

	fprintf(receiveData, "cudaStatus = cudaGetLastError(); if (cudaStatus != cudaSuccess) {fprintf(stderr, \"addKernel launch failed: %%s\\n\", cudaGetErrorString(cudaStatus));goto Error;}\n\n");
	fprintf(receiveData, "cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) {fprintf(stderr, \"cudaDeviceSynchronize returned error code %%d after launching addKernel!\\n\", cudaStatus);goto Error;}");

	fprintf(freeDevice, "Error:\n");

	for (int i = 0; i < Model_getNumSpecies(m); i++) {
		Species_t* s = ListOf_get(species, i);
		Key = Species_getId(s);
		Value = Species_getInitialAmount(s);
		fprintf(initializeSpecies, "float %s = %lf;\n", Key, Value);
		fprintf(initializeSpecies, "float* dev_%s = 0;\n", Key);
		fprintf(initializeSpecies, "cudaStatus = cudaMalloc(&dev_%s, sizeof(float));\n", Key);
		fprintf(initializeSpecies, "if (cudaStatus != cudaSuccess) {fprintf(stderr, \"cudaMalloc failed!\");goto Error;}\n");
		fprintf(initializeSpecies, "cudaStatus = cudaMemcpy(dev_%s, &%s, sizeof(float), cudaMemcpyHostToDevice);\n", Key, Key);
		fprintf(initializeSpecies, "if (cudaStatus != cudaSuccess) {fprintf(stderr, \"cudaMemcpy failed!\");goto Error;}\n");

		//at first initializing the regular species (without the aux suffix) looks not optimal or weird,
		//but it saves some time avoiding the need to open a new FILE just to avoid doing another loop
		fprintf(kernelFunction, ", float %s, float* %s_aux", Key, Key);

		//initialized with 0 because it does not matter as it will be overwritten each loop
		fprintf(kernelCall, ", 0, dev_%s", Key, Key);

		fprintf(receiveData, "cudaStatus = cudaMemcpy(&%s, dev_%s, sizeof(float), cudaMemcpyDeviceToHost);\n", Key, Key);
		fprintf(receiveData, "if (cudaStatus != cudaSuccess) {fprintf(stderr, \"cudaMemcpy failed!\");goto Error;}\n");

		fprintf(freeDevice, "cudaFree(dev_%s);\n", Key);

		fprintf(printResults, "printf(\"%s = %%f\\n\", %s);\n", Key, Key);

		fprintf(kernelVariablesInit, "%s = *%s_aux;\n", Key, Key);

	}

	fprintf(kernelCall, ");\n\n");
	fprintf(receiveData, "cudaStatus = cudaGetLastError(); if (cudaStatus != cudaSuccess) {fprintf(stderr, \"addKernel launch failed: %%s\\n\", cudaGetErrorString(cudaStatus));goto Error;}\n\n");
	fprintf(receiveData, "cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) {fprintf(stderr, \"cudaDeviceSynchronize returned error code %%d after launching addKernel!\\n\", cudaStatus);goto Error;}");


	fprintf(kernelFunction, ") {\n");
	fprintf(kernelFunction, "float reactionRate;\n");
	fprintf(kernelFunction, "for(int i = 0; i < numSimulations; i++){\n");

	fprintf(kernelVariablesInit, "simulateStepReaction(threadIdx.x);\n");
	fprintf(kernelVariablesInit, "__syncthreads();\n");
	fprintf(kernelVariablesInit, "}\n}\n");

	for (int i = 0; i < Model_getNumCompartments(m); i++) {
		Compartment_t* c = ListOf_get(compartments, i);
		Key = Compartment_getId(c);
		Value = Compartment_getSize(c);
		fprintf(defineConstants, "#define %s %lf\n", Key, Value);
	}

	for (int i = 0; i < Model_getNumParameters(m); i++) {
		Parameter_t* p = ListOf_get(parameters, i);
		Key = Parameter_getId(p);
		Value = Parameter_getValue(p);
		fprintf(defineConstants, "#define %s %lf\n", Key, Value);
	}

	int streams_closed = fcloseall();

	if (streams_closed == EOF) {
		printf("Error closing files!");
		exit(1);
	}
}


int
main(int argc, char *argv[])
{
	SBMLDocument_t *d;
	Model_t        *m;
	double step;
	int simulations;

	if (argc != 4)
	{
		printf("Usage: generateCUDA filename stepSize numberOfSimulations\n");
		return 1;
	}

	d = readSBML(argv[1]);
	m = SBMLDocument_getModel(d);

	step = atof(argv[2]);
	simulations = atoi(argv[3]);

	SBMLDocument_printErrors(d, stdout);

	generateCUDA(m, step, simulations);

	SBMLDocument_free(d);
	return 0;
}
