
#include <stdio.h>
#include <stdlib.h>

#include <sbml/SBMLTypes.h>

#include <math.h>
#include <stddef.h>
#include <string.h>

#define SQR(x) ((x)*(x))
#define SQRT(x) pow((x),(.5))

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

	fprintf(defineSpeciesUpdates, "#define speciesUpdate(i) \\\n");
	fprintf(defineSpeciesUpdates, "switch (i) { \\\n");

	//get list of reactions and formulas
	KineticLaw_t *kl;

	for (int i = 0; i < Model_getNumReactions(m); i++)
	{
		//save formulas processing time vs memory size
		if (Reaction_isSetKineticLaw(Model_getReaction(m, i)))
		{
			kl = Reaction_getKineticLaw(Model_getReaction(m, i));
			if (KineticLaw_isSetMath(kl))
			{
				fprintf(updatePropencities, "p[%d] = %s; \n", i, SBML_formulaToString(KineticLaw_getMath(kl)));

				fprintf(defineSpeciesUpdates, "case %d: \\\n", i);

				//reactants
				ListOf_t* reactants = Reaction_getListOfReactants(Model_getReaction(m, i));
				for (int k = 0; k < Reaction_getNumReactants(Model_getReaction(m, i)); k++) {
					SpeciesReference_t* reactant = ListOf_get(reactants, k);
					if (Species_getConstant(ListOfSpecies_getById(species, SpeciesReference_getSpecies(reactant))) == 0) {
						fprintf(defineSpeciesUpdates, "%s -= %lf; \\\n", SpeciesReference_getSpecies(reactant), SpeciesReference_getStoichiometry(reactant));
					}
				}

				//products
				ListOf_t* products = Reaction_getListOfProducts(Model_getReaction(m, i));
				for (int k = 0; k < Reaction_getNumProducts(Model_getReaction(m, i)); k++) {
					SpeciesReference_t* product = ListOf_get(products, k);
					if (Species_getConstant(ListOfSpecies_getById(species, SpeciesReference_getSpecies(product))) == 0) {
						fprintf(defineSpeciesUpdates, "%s += %lf; \\\n", SpeciesReference_getSpecies(product), SpeciesReference_getStoichiometry(product));
					}
				}

				fprintf(defineSpeciesUpdates, "break; \\\n");


				for (int j = 0; j < KineticLaw_getNumParameters(kl); j++) {
					Parameter_t* p = KineticLaw_getParameter(kl, j);
					fprintf(defineConstants, "#define %s %lf\n", Parameter_getId(p), Parameter_getValue(p));
				}
			}
		}
	}

	fprintf(defineSpeciesUpdates, "} \\\n");

	//get initial values and constants

	fprintf(kernelFunction, "\n__global__ \n");
	fprintf(kernelFunction, "void simulate (float* output, curandState *state, float step, float endTime");

	fprintf(initializeSpecies, "cudaError_t cudaStatus;\n");

	fprintf(initializeSpecies, "float* output;\n");
	fprintf(initializeSpecies, "float* dev_output;\n");

	fprintf(initializeSpecies, "output = (float*)malloc(%d*%d*sizeof(float));\n", (int)ceil(endTime / step), Model_getNumSpecies(m));

	fprintf(initializeSpecies, "for(int i = 0; i < %d*%d; i++){\n", (int)ceil(endTime / step), Model_getNumSpecies(m));
	fprintf(initializeSpecies, "output[i] = 0;\n");
	fprintf(initializeSpecies, "}\n");

	fprintf(initializeSpecies, "cudaStatus = cudaMalloc(&dev_output, %d*%d*sizeof(float));\n", (int)ceil(endTime/step), Model_getNumSpecies(m));
	fprintf(initializeSpecies, "if (cudaStatus != cudaSuccess) {fprintf(stderr, \"cudaMalloc failed!\");goto Error;}\n");
	fprintf(initializeSpecies, "cudaStatus = cudaMemcpy(dev_output, output, %d*%d*sizeof(float), cudaMemcpyHostToDevice);\n", (int)ceil(endTime/step), Model_getNumSpecies(m));
	fprintf(initializeSpecies, "if (cudaStatus != cudaSuccess) {fprintf(stderr, \"cudaMemcpy failed!\");goto Error;}\n");

	fprintf(kernelCall, "simulate<<<1, %d>>>(dev_output, devStates, %lf, %lf", simulations, step, endTime);

	fprintf(receiveData, "cudaStatus = cudaGetLastError(); if (cudaStatus != cudaSuccess) {fprintf(stderr, \"addKernel launch failed: %%s\\n\", cudaGetErrorString(cudaStatus));goto Error;}\n\n");
	fprintf(receiveData, "cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) {fprintf(stderr, \"cudaDeviceSynchronize returned error code %%d after launching addKernel!\\n\", cudaStatus);goto Error;}");

	fprintf(receiveData, "\n\ncudaStatus = cudaMemcpy(output, dev_output, %d*%d*sizeof(float), cudaMemcpyDeviceToHost);\n", (int)ceil(endTime / step), Model_getNumSpecies(m));
	fprintf(receiveData, "if (cudaStatus != cudaSuccess) {fprintf(stderr, \"cudaMemcpy failed!\");goto Error;}\n");

	fprintf(freeDevice, "Error:\n");
	fprintf(freeDevice, "cudaFree(dev_output);\n");

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

		fprintf(kernelFunction, ", float* %s_aux", Key);

		fprintf(kernelCall, ", dev_%s", Key);

		fprintf(receiveData, "cudaStatus = cudaMemcpy(&%s, dev_%s, sizeof(float), cudaMemcpyDeviceToHost);\n", Key, Key);
		fprintf(receiveData, "if (cudaStatus != cudaSuccess) {fprintf(stderr, \"cudaMemcpy failed!\");goto Error;}\n");

		fprintf(freeDevice, "cudaFree(dev_%s);\n", Key);

		fprintf(kernelVariablesInit, "float %s = *%s_aux;\n", Key, Key);

	}

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

	fprintf(exportResults, "for(int i = 0; i < %d; i++){\n", (int)ceil(endTime/step));
	fprintf(exportResults, "fprintf(results, \"%%lf\", %lf*i);\n", step);
	fprintf(exportResults, "for(int j = 0; j < %d; j++){\n", Model_getNumSpecies(m));
	fprintf(exportResults, "fprintf(results, \", %%lf\", output[%d*i+j]/%d);\n", Model_getNumSpecies(m), simulations);
	fprintf(exportResults, "}\n");
	fprintf(exportResults, "fprintf(results, \"\\n\");\n");
	fprintf(exportResults, "}\n");
	fprintf(exportResults, "fprintf(results, \"\\n\");\n");

	fprintf(kernelCall, ");\n\n");
	fprintf(receiveData, "cudaStatus = cudaGetLastError(); if (cudaStatus != cudaSuccess) {fprintf(stderr, \"addKernel launch failed: %%s\\n\", cudaGetErrorString(cudaStatus));goto Error;}\n\n");
	fprintf(receiveData, "cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) {fprintf(stderr, \"cudaDeviceSynchronize returned error code %%d after launching addKernel!\\n\", cudaStatus);goto Error;}");


	fprintf(kernelFunction, ") {\n");
	fprintf(kernelFunction, "int reaction, stepCount = 0;\n");
	fprintf(kernelFunction, "float time = 0;\n");
	fprintf(kernelFunction, "float sum_p, sum_p_aux, timeStep, random;\n");
	fprintf(kernelFunction, "float p[%d];\n", Model_getNumReactions(m));

	fprintf(kernelVariablesInit, "while(time < endTime){\n");

	fprintf(updatePropencities, "if(time >= step * stepCount){\n");

	for (int i = 0; i < Model_getNumSpecies(m); i++) {
		fprintf(updatePropencities, "atomicAdd(&output[stepCount*%d + %d], %s);\n", Model_getNumSpecies(m), i, Species_getId(ListOf_get(species, i)));
	}

	fprintf(updatePropencities, "stepCount++;\n");
	fprintf(updatePropencities, "}\n");



	fprintf(updatePropencities, "sum_p = 0;\n");
	fprintf(updatePropencities, "for(int i = 0; i < %d; i++){\n", Model_getNumReactions(m));
	fprintf(updatePropencities, "sum_p += p[i];\n");
	fprintf(updatePropencities, "}\n");

	fprintf(updatePropencities, "curandState localState = state[threadIdx.x];\n");
	fprintf(updatePropencities, "random = curand_uniform(&localState);\n");

	fprintf(updatePropencities, "if(sum_p > 0) timeStep = -log(random)/sum_p;\n");
	fprintf(updatePropencities, "else break;\n");

	fprintf(updatePropencities, "random = curand_uniform(&localState);\n");

	fprintf(updatePropencities, "reaction = -1;\n");
	fprintf(updatePropencities, "sum_p_aux = 0;\n");
	fprintf(updatePropencities, "random *= sum_p;\n");
	fprintf(updatePropencities, "for(int i = 0; i < %d; i++){\n", Model_getNumReactions(m));
	fprintf(updatePropencities, "sum_p_aux += p[i];\n");
	fprintf(updatePropencities, "if(random < sum_p_aux){\n");
	fprintf(updatePropencities, "reaction = i;\n");
	fprintf(updatePropencities, "break;\n");
	fprintf(updatePropencities, "}\n}\n");

	fprintf(updatePropencities, "speciesUpdate(reaction);\n");

	ListOf_t* events = Model_getListOfEvents(m);
	for (int i = 0; i < Model_getNumEvents(m); i++) {
		Event_t* event = ListOf_get(events, i);
		fprintf(kernelFunction, "int triggerEvent%d = 0;\n", i);
		fprintf(updatePropencities, "if(triggerEvent%d == 0 && %s){\n", i, SBML_formulaToL3String(Trigger_getMath(Event_getTrigger(event))));
		fprintf(updatePropencities, "triggerEvent%d = 1;\n", i);
		ListOf_t* eventAssignments = Event_getListOfEventAssignments(event);
		for (int j = 0; j < Event_getNumEventAssignments(event); j++) {
			fprintf(updatePropencities, "%s += %s;\n", EventAssignment_getVariable(ListOf_get(eventAssignments, j)), SBML_formulaToL3String(EventAssignment_getMath(ListOf_get(eventAssignments, j))));
		}
		fprintf(updatePropencities, "}\n");
	}

	fprintf(updatePropencities, "time += timeStep;\n");


	fprintf(updatePropencities, "}\n}\n");

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
