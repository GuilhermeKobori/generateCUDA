
#include <stdio.h>
#include <stdlib.h>

#include <sbml/SBMLTypes.h>

#include <math.h>
#include <stddef.h>
#include <string.h>

#define SQR(x) ((x)*(x))
#define SQRT(x) pow((x),(.5))

#define SEGMENT_SIZE 2000

void generateCUDA(Model_t* m, double step, int simulations, double endTime) {
	//get compartments and initial concentrations in key value pairs
	ListOf_t* species = Model_getListOfSpecies(m);
	ListOf_t* compartments = Model_getListOfCompartments(m);
	ListOf_t* parameters = Model_getListOfParameters(m);
	char* Key;
	double Value;
	FILE* updatePropencities = fopen("GILLupdatePropencities", "w");		//contains metaprogramming for updating the propensities values
	FILE* defineConstants = fopen("GILLdefineConstants", "w");				//contains constants for the simulation, containers, local parameters and global paramenter
	FILE* initializeSpecies = fopen("GILLinitializeSpecies", "w");			//declares a variable and associates the value for each species, also creates and initializes devices variables
	FILE* kernelFunction = fopen("GILLkernelFunction", "w");				//device function, loops around a simple Runge-Kutta simulation
	FILE* kernelCall = fopen("GILLkernelCall", "w");						//declares the call for the kernel function
	FILE* kernelVariablesInit = fopen("GILLkernelVariablesInit", "w");		//initializes the variables for the kernel function
	FILE* kernelWriteInGlobal = fopen("GILLkernelWriteInGlobal", "w");		//writes the variables in global memory for next kernel execution
	FILE* receiveData = fopen("GILLreceiveData", "w");						//contains code that receives data from device to host
	FILE* freeDevice = fopen("GILLfreeDevice", "w");						//contains free declarations for the device variables
	FILE* exportResults = fopen("GILLexportResults", "w");					//prints each species with the values received from device
	FILE* curandInit = fopen("GILLcurandInit", "w");						//initializes random number generator
	FILE* curandCall = fopen("GILLcurandCall", "w");						//calls random number generator initializer
	FILE* defineSpeciesUpdates = fopen("GILLdefineSpeciesUpdates", "w");	//defines values for each species that must be updated after a reaction occurs

	//error handling
	if (updatePropencities == NULL)
	{
		printf("Error accessing updatePropencities!");
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
	if (kernelVariablesInit == NULL)
	{
		printf("Error accessing kernelVariablesInit!");
		exit(1);
	}
	if (kernelWriteInGlobal == NULL)
	{
		printf("Error accessing kernelWriteInGlobal!");
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
	if (exportResults == NULL)
	{
		printf("Error accessing exportResults!");
		exit(1);
	}
	if (curandInit == NULL)
	{
		printf("Error accessing curandInit!");
		exit(1);
	}
	if (curandCall == NULL)
	{
		printf("Error accessing curandCall!");
		exit(1);
	}
	if (defineSpeciesUpdates == NULL)
	{
		printf("Error accessing defineSpeciesUpdates!");
		exit(1);
	}

	fprintf(curandInit, "\n__global__ \nvoid initCurand(curandState* state, unsigned long long seed){\n");
	fprintf(curandInit, "curand_init(seed, threadIdx.x, 0, &state[threadIdx.x]);\n");
	fprintf(curandInit, "}\n");

	fprintf(curandCall, "curandState *devStates;\n");
	fprintf(curandCall, "CUDA_CALL(cudaMalloc((void **)&devStates, %d * sizeof(curandState)));\n", simulations);
	fprintf(curandCall, "initCurand<<<1, %d>>>(devStates, SEED);\n", simulations);

	int maxReactionSpecies = 0;
	for (int i = 0; i < Model_getNumReactions(m); i++)
	{
		if (maxReactionSpecies < Reaction_getNumReactants(Model_getReaction(m, i)) + Reaction_getNumProducts(Model_getReaction(m, i))) {
			maxReactionSpecies = Reaction_getNumReactants(Model_getReaction(m, i)) + Reaction_getNumProducts(Model_getReaction(m, i));
		}
	}

	fprintf(defineSpeciesUpdates, "int reactionsSpecies[%d][%d];\nint reactionsValues[%d][%d];\n", Model_getNumReactions(m), maxReactionSpecies, Model_getNumReactions(m), maxReactionSpecies);
	fprintf(defineSpeciesUpdates, "for(int i = 0; i < %d; i++){\n", Model_getNumReactions(m));
	fprintf(defineSpeciesUpdates, "for(int j = 0; j < %d; j++){\n", maxReactionSpecies);
	fprintf(defineSpeciesUpdates, "reactionsSpecies[i][j] = -1;\n");
	fprintf(defineSpeciesUpdates, "reactionsValues[i][j] = 0;\n");
	fprintf(defineSpeciesUpdates, "}\n");
	fprintf(defineSpeciesUpdates, "}\n");

	//get list of reactions and formulas
	KineticLaw_t *kl;

	int index;
	for (int i = 0; i < Model_getNumReactions(m); i++)
	{
		index = 0;
		//save formulas processing time vs memory size
		if (Reaction_isSetKineticLaw(Model_getReaction(m, i)))
		{
			kl = Reaction_getKineticLaw(Model_getReaction(m, i));
			if (KineticLaw_isSetMath(kl))
			{
				if (i == 0) fprintf(updatePropencities, "cummulative_p[%d] = %s; \n", i, SBML_formulaToString(KineticLaw_getMath(kl)));
				else fprintf(updatePropencities, "cummulative_p[%d] = cummulative_p[%d] + %s; \n", i, i - 1, SBML_formulaToString(KineticLaw_getMath(kl)));

				//reactants
				ListOf_t* reactants = Reaction_getListOfReactants(Model_getReaction(m, i));
				for (int k = 0; k < Reaction_getNumReactants(Model_getReaction(m, i)); k++) {
					SpeciesReference_t* reactant = ListOf_get(reactants, k);
					if (Species_getConstant(ListOfSpecies_getById(species, SpeciesReference_getSpecies(reactant))) == 0) {
						fprintf(defineSpeciesUpdates, "reactionsSpecies[%d][%d] = %s_id; \n", i, index, SpeciesReference_getSpecies(reactant));
						fprintf(defineSpeciesUpdates, "reactionsValues[%d][%d] = -%.10lf; \n", i, index, SpeciesReference_getStoichiometry(reactant));
						index++;
					}
				}

				//products
				ListOf_t* products = Reaction_getListOfProducts(Model_getReaction(m, i));
				for (int k = 0; k < Reaction_getNumProducts(Model_getReaction(m, i)); k++) {
					SpeciesReference_t* product = ListOf_get(products, k);
					if (Species_getConstant(ListOfSpecies_getById(species, SpeciesReference_getSpecies(product))) == 0) {
						fprintf(defineSpeciesUpdates, "reactionsSpecies[%d][%d] = %s_id; \n", i, index, SpeciesReference_getSpecies(product));
						fprintf(defineSpeciesUpdates, "reactionsValues[%d][%d] = %.10lf; \n", i, index, SpeciesReference_getStoichiometry(product));
						index++;
					}
				}

				for (int j = 0; j < KineticLaw_getNumParameters(kl); j++) {
					Parameter_t* p = KineticLaw_getParameter(kl, j);
					fprintf(defineConstants, "#define %s %.10lf\n", Parameter_getId(p), Parameter_getValue(p));
				}
			}
		}
	}

	fprintf(defineSpeciesUpdates, "curandState localState = state[threadIdx.x];\n");
	fprintf(defineSpeciesUpdates, "while(time < endTime && time < (numberOfExecutions + 1)*segmentSize){\n");

	//get initial values and constants

	fprintf(kernelFunction, "\n__global__ \n");
	fprintf(kernelFunction, "void simulate (int numberOfExecutions, float* output, curandState *state, float step, float endTime, float segmentSize, float* species_global");

	fprintf(initializeSpecies, "cudaError_t cudaStatus;\n");

	fprintf(initializeSpecies, "float* output;\n");
	fprintf(initializeSpecies, "float* dev_output;\n");

	fprintf(initializeSpecies, "output = (float*)malloc(%d*%d*sizeof(float));\n", (int)ceil(endTime / step), Model_getNumSpecies(m));

	fprintf(initializeSpecies, "for(int i = 0; i < %d*%d; i++){\n", (int)ceil(endTime / step), Model_getNumSpecies(m));
	fprintf(initializeSpecies, "output[i] = 0;\n");
	fprintf(initializeSpecies, "}\n");

	fprintf(initializeSpecies, "cudaStatus = cudaMalloc(&dev_output, %d*%d*sizeof(float));\n", (int)ceil(endTime / step), Model_getNumSpecies(m));
	fprintf(initializeSpecies, "if (cudaStatus != cudaSuccess) {fprintf(stderr, \"cudaMalloc failed!\");goto Error;}\n");
	fprintf(initializeSpecies, "cudaStatus = cudaMemcpy(dev_output, output, %d*%d*sizeof(float), cudaMemcpyHostToDevice);\n", (int)ceil(endTime / step), Model_getNumSpecies(m));
	fprintf(initializeSpecies, "if (cudaStatus != cudaSuccess) {fprintf(stderr, \"cudaMemcpy failed!\");goto Error;}\n");

	fprintf(initializeSpecies, "float* species_global;\n");
	fprintf(initializeSpecies, "cudaStatus = cudaMalloc(&species_global, %d*%d*sizeof(float));\n", Model_getNumSpecies(m), simulations);
	fprintf(initializeSpecies, "if (cudaStatus != cudaSuccess) {fprintf(stderr, \"cudaMalloc failed!\");goto Error;}\n");

	fprintf(kernelCall, "for(int i = 0; i < %d; i++){\n", (int)ceil(endTime / SEGMENT_SIZE));
	//This feels weird TODO invenstigate better ways?
	int segmentSize = SEGMENT_SIZE;
	fprintf(kernelCall, "simulate<<<1, %d>>>(i, dev_output, devStates, %.10lf, %.10lf, %d, species_global", simulations, step, endTime, segmentSize);

	fprintf(receiveData, "\n\ncudaStatus = cudaMemcpy(output, dev_output, %d*%d*sizeof(float), cudaMemcpyDeviceToHost);\n", (int)ceil(endTime / step), Model_getNumSpecies(m));
	fprintf(receiveData, "if (cudaStatus != cudaSuccess) {fprintf(stderr, \"cudaMemcpy failed!\");goto Error;}\n");

	fprintf(freeDevice, "Error:\n");
	fprintf(freeDevice, "cudaFree(dev_output);\n");
	fprintf(freeDevice, "cudaFree(species_global);\n");


	fprintf(kernelVariablesInit, "float species[%d];\n", Model_getNumSpecies(m));

	for (int i = 0; i < Model_getNumSpecies(m); i++) {
		Species_t* s = ListOf_get(species, i);
		Key = Species_getId(s);
		Value = Species_getInitialAmount(s);
		fprintf(initializeSpecies, "float host_%s = %.10lf;\n", Key, Value);
		fprintf(initializeSpecies, "float* dev_%s;\n", Key);
		fprintf(initializeSpecies, "cudaStatus = cudaMalloc(&dev_%s, sizeof(float));\n", Key);
		fprintf(initializeSpecies, "if (cudaStatus != cudaSuccess) {fprintf(stderr, \"cudaMalloc failed!\");goto Error;}\n");
		fprintf(initializeSpecies, "cudaStatus = cudaMemcpy(dev_%s, &host_%s, sizeof(float), cudaMemcpyHostToDevice);\n", Key, Key);
		fprintf(initializeSpecies, "if (cudaStatus != cudaSuccess) {fprintf(stderr, \"cudaMemcpy failed!\");goto Error;}\n");

		fprintf(kernelFunction, ", float* %s_aux", Key);

		fprintf(kernelCall, ", dev_%s", Key);

		fprintf(receiveData, "cudaStatus = cudaMemcpy(&host_%s, dev_%s, sizeof(float), cudaMemcpyDeviceToHost);\n", Key, Key);
		fprintf(receiveData, "if (cudaStatus != cudaSuccess) {fprintf(stderr, \"cudaMemcpy failed!\");goto Error;}\n");

		fprintf(freeDevice, "cudaFree(dev_%s);\n", Key);

		fprintf(kernelVariablesInit, "if(numberOfExecutions == 0){\n");
		fprintf(kernelVariablesInit, "species[%d] = *%s_aux;\n", i, Key);
		fprintf(kernelVariablesInit, "} else {\n");
		fprintf(kernelVariablesInit, "species[%d] = species_global[%s_id*%d + threadIdx.x];\n", i, Key, simulations);
		fprintf(kernelVariablesInit, "}\n");

		fprintf(kernelWriteInGlobal, "species_global[%s_id*%d + threadIdx.x] = species[%d];\n", Key, simulations, i);

		fprintf(defineConstants, "#define %s species[%d]\n", Key, i);
		fprintf(defineConstants, "#define %s_id %d\n", Key, i);
	}


	fprintf(kernelWriteInGlobal, "state[threadIdx.x] = localState;\n");
	fprintf(kernelWriteInGlobal, "}\n");

	fprintf(exportResults, "FILE* results = fopen(\"results.csv\", \"w\");\n");
	fprintf(exportResults, "if(results == NULL){\n");
	fprintf(exportResults, "printf(\"Error acesssing results!\");\n");
	fprintf(exportResults, "exit(1);\n");
	fprintf(exportResults, "}\n");

	fprintf(exportResults, "fprintf(results, \"time\");\n");

	for (int i = 0; i < Model_getNumSpecies(m); i++) {
		fprintf(exportResults, "fprintf(results, \", %s\");\n", Species_getId(ListOf_get(species, i)));
	}

	fprintf(exportResults, "fprintf(results, \"\\n\");\n");

	fprintf(exportResults, "for(int i = 0; i < %d; i++){\n", (int)ceil(endTime / step));
	fprintf(exportResults, "fprintf(results, \"%%.10lf\", %.10lf*i);\n", step);
	fprintf(exportResults, "for(int j = 0; j < %d; j++){\n", Model_getNumSpecies(m));
	fprintf(exportResults, "fprintf(results, \", %%.10lf\", output[%d*i+j]/%d);\n", Model_getNumSpecies(m), simulations);
	fprintf(exportResults, "}\n");
	fprintf(exportResults, "fprintf(results, \"\\n\");\n");
	fprintf(exportResults, "}\n");
	fprintf(exportResults, "fprintf(results, \"\\n\");\n");

	fprintf(kernelCall, ");\n\n");
	fprintf(kernelCall, "cudaStatus = cudaGetLastError(); if (cudaStatus != cudaSuccess) {fprintf(stderr, \"addKernel launch failed: %%s\\n\", cudaGetErrorString(cudaStatus));goto Error;}\n\n");
	fprintf(kernelCall, "cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) {fprintf(stderr, \"cudaDeviceSynchronize returned error code %%d after launching addKernel!\\n\", cudaStatus);goto Error;}");
	fprintf(kernelCall, "}\n");

	fprintf(kernelFunction, ") {\n");
	fprintf(kernelFunction, "int reaction, stepCount = 0;\n");
	fprintf(kernelFunction, "int indexMin, indexMax;\n");
	fprintf(kernelFunction, "float time = numberOfExecutions*segmentSize;\n");
	fprintf(kernelFunction, "float sum_p, timeStep, random;\n");
	fprintf(kernelFunction, "float cummulative_p[%d];\n", Model_getNumReactions(m));

	fprintf(updatePropencities, "if(time >= segmentSize * numberOfExecutions + step * stepCount){\n");

	for (int i = 0; i < Model_getNumSpecies(m); i++) {
		fprintf(updatePropencities, "atomicAdd(&output[%d*%d*numberOfExecutions + stepCount*%d + %d], species[%d]);\n", Model_getNumSpecies(m), (int)ceil(segmentSize / step), Model_getNumSpecies(m), i, i);
	}

	fprintf(updatePropencities, "stepCount++;\n");
	fprintf(updatePropencities, "}\n");

	fprintf(updatePropencities, "sum_p = cummulative_p[%d];\n", Model_getNumReactions(m) - 1);


	fprintf(updatePropencities, "random = curand_uniform(&localState);\n");

	fprintf(updatePropencities, "if(sum_p > 0) timeStep = -log(random)/sum_p;\n");
	fprintf(updatePropencities, "else break;\n");

	fprintf(updatePropencities, "random = curand_uniform(&localState);\n");

	fprintf(updatePropencities, "random *= sum_p;\n");
	fprintf(updatePropencities, "indexMin = 0;\n");
	fprintf(updatePropencities, "indexMax = %d;\n", Model_getNumReactions(m) - 1);
	fprintf(updatePropencities, "while(indexMax > indexMin){\n");
	fprintf(updatePropencities, "reaction = (indexMin + indexMax)/2;\n");
	fprintf(updatePropencities, "if(cummulative_p[reaction - 1] <= random){\n");
	fprintf(updatePropencities, "if(cummulative_p[reaction] > random){\n");
	fprintf(updatePropencities, "break;\n");
	fprintf(updatePropencities, "}\n");
	fprintf(updatePropencities, "else{\n");
	fprintf(updatePropencities, "indexMin = reaction;\n");
	fprintf(updatePropencities, "}\n");
	fprintf(updatePropencities, "}\n");
	fprintf(updatePropencities, "else{\n");
	fprintf(updatePropencities, "indexMax = reaction;\n");
	fprintf(updatePropencities, "}\n");
	fprintf(updatePropencities, "}\n");

	fprintf(updatePropencities, "for(int i = 0; i < %d; i++){\n", maxReactionSpecies);
	fprintf(updatePropencities, "if(reactionsSpecies[reaction][i] == -1) {break;}\n");
	fprintf(updatePropencities, "species[reactionsSpecies[reaction][i]] += reactionsValues[reaction][i];\n");
	fprintf(updatePropencities, "}\n");

	ListOf_t* events = Model_getListOfEvents(m);
	for (int i = 0; i < Model_getNumEvents(m); i++) {
		Event_t* event = ListOf_get(events, i);
		fprintf(kernelFunction, "int triggerEvent%d = 0;\n", i);
		fprintf(kernelFunction, "if(%s) {triggerEvent%d = 1;}\n", SBML_formulaToL3String(Trigger_getMath(Event_getTrigger(event))), i);
		fprintf(updatePropencities, "if(triggerEvent%d == 0 && %s){\n", i, SBML_formulaToL3String(Trigger_getMath(Event_getTrigger(event))));
		fprintf(updatePropencities, "triggerEvent%d = 1;\n", i);
		ListOf_t* eventAssignments = Event_getListOfEventAssignments(event);
		for (int j = 0; j < Event_getNumEventAssignments(event); j++) {
			fprintf(updatePropencities, "%s += %s;\n", EventAssignment_getVariable(ListOf_get(eventAssignments, j)), SBML_formulaToL3String(EventAssignment_getMath(ListOf_get(eventAssignments, j))));
		}
		fprintf(updatePropencities, "}\n");
	}

	fprintf(updatePropencities, "time += timeStep;\n");


	fprintf(updatePropencities, "}\n");

	for (int i = 0; i < Model_getNumCompartments(m); i++) {
		Compartment_t* c = ListOf_get(compartments, i);
		Key = Compartment_getId(c);
		Value = Compartment_getSize(c);
		fprintf(defineConstants, "#define %s %.10lf\n", Key, Value);
	}

	for (int i = 0; i < Model_getNumParameters(m); i++) {
		Parameter_t* p = ListOf_get(parameters, i);
		Key = Parameter_getId(p);
		Value = Parameter_getValue(p);
		fprintf(defineConstants, "#define %s %.10lf\n", Key, Value);
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
	double step, endTime;
	int numSimulations;

	if (argc != 5)
	{
		printf("Usage: generateCUDA filename sampleTimeStep endTime numberKernels\n");
		return 1;
	}

	d = readSBML(argv[1]);
	m = SBMLDocument_getModel(d);

	step = atof(argv[2]);
	endTime = atof(argv[3]);
	numSimulations = atoi(argv[4]);

	SBMLDocument_printErrors(d, stdout);

	generateCUDA(m, step, numSimulations, endTime);

	SBMLDocument_free(d);
	return 0;
}
