#include "neural_network.h"
#define TILE_WIDTH 16
#define INF 1000000000.0f
//Basic cuda error checking macro
//TODO: Add cuRAND and cuBLAS error checking macros
//TODO: Wrap all calls in relevant error checking macros
#define cudaCheckError()\
{\
	cudaError_t e = cudaGetLastError();\
	if(e != cudaSuccess)\
	{\
		printf("CUDA failure: %s%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));\
		exit(EXIT_FAILURE);\
	}\
}

using namespace std;

// Normalizes a vector to [-MaxValue, MaxValue]
__global__
void Normalize(float *Array, int Number, float MaxValue)
{
    int Index = blockDim.x * blockIdx.x + threadIdx.x;
    if(Index < Number)
        Array[Index] = 2 * (Array[Index] - 0.5f) * MaxValue;
}

// Transpose a matrix
__global__
void Transpose(float *InputMatrix, float *OutputMatrix, int Rows, int Columns)
{
	int IdX = blockDim.x * blockIdx.x + threadIdx.x;
	int IdY = blockDim.y * blockIdx.y + threadIdx.y;
	int TX = threadIdx.x;
	int TY = threadIdx.y;

	__shared__ float Tile[TILE_WIDTH][TILE_WIDTH];

	if(IdX < Columns && IdY < Rows)
	{
		Tile[TX][TY] = InputMatrix[IdX + Columns * IdY];
		OutputMatrix[IdY + Rows * IdX] = Tile[TX][TY];
	}
}

// Initialize to infinity
__global__
void InitToVal(float *Input, int Size, float Value)
{
	int Index = blockDim.x * blockIdx.x + threadIdx.x;
	if(Index < Size)
		Input[Index] = Value;
}

// Small kernel for device to device memory transfers
__global__
void DeviceToDevice(float *Destination, float *Source, int Size)
{
	int Index = blockDim.x * blockIdx.x + threadIdx.x;
	if(Index < Size)
		Destination[Index] = Source[Index];
}

// Initialize random number generator states
__global__
void InitRNGStates(curandState_t *States, int Size)
{
	int Index = blockDim.x * blockIdx.x + threadIdx.x;
	if(Index < Size)
		curand_init(Index, Index, Index, &States[Index]);
}

// LeakyReLU activation function
__global__
void LeakyReLU(float *Input, int Size)
{
	int Index = blockDim.x * blockIdx.x + threadIdx.x;
	if(Index < Size)
	{
		if(Input[Index] < 0.0f)
			Input[Index] = 0.001 * Input[Index];
	}
}

__global__
void Sigmoid(float *Input, int Size)
{
	int Index = blockDim.x * blockIdx.x + threadIdx.x;
	if(Index < Size)
    	Input[Index] = (1 / (1 + __expf(-Input[Index])));
}

// Kernel which actually trains the data.
__global__
void FeedForward(NNParameters *NNP, PSOParameters *PSOP)
{
    int Index = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ NNParameters NNParams;
	__shared__ PSOParameters PSOParams;

	if(threadIdx.x == 0)
	{
		NNParams = *NNP;
		PSOParams = *PSOP;
	}

	if(Index < PSOParams.NumParticles)
	{
		//Pointer to weights and biases
		float *WeightsAndBiases = &NNParams.WeightsAndBiases[Index * NNParams.NetworkSize];

		//Input, output, matrix and temporary pointers
		float *Input;
		float *Output;
		float *Matrix;
		float *Temp;

		//Fitness value
	    float Fitness = 0.0f;

		//cuBLAS handle initialization
		cublasHandle_t Handle;
		cublasCreate(&Handle);

		//Alpha and beta values
		float Alpha = 1.0f;
	    float Beta = 0.0f;

		Fitness = 0.0f;

		//Main feed forward work to be done here
		//Calculate fitness, i.e. loss (MSE?)
        for(int j = 0; j < NNParams.NumVectors; j++)
		{
			//Input hidden multiplication + biases
			Input = &(NNParams.InputFeatures[NNParams.InputNeurons * j]);
			Output = &(NNParams.IntermediateIO[NNParams.MaxIOLength * Index]);
			Matrix = &(NNParams.WeightsAndBiases[NNParams.NetworkSize * Index]);

			cublasSgemv(Handle, CUBLAS_OP_N,
				NNParams.HiddenNeurons, NNParams.InputNeurons, &Alpha,
				Matrix, NNParams.HiddenNeurons, Input, 1, &Beta, Output, 1);
			cudaDeviceSynchronize();

			Matrix += NNParams.InputNeurons * NNParams.HiddenNeurons;

			//Add biases
			cublasSaxpy(Handle, NNParams.HiddenNeurons,
				&Alpha, Matrix, 1, Output, 1);

			//Activation function
			LeakyReLU <<<(NNParams.HiddenNeurons - 1) / 32 + 1, 32>>> (Output, NNParams.HiddenNeurons);
			cudaDeviceSynchronize();

			Input = Output + NNParams.MaxIOLength / 2;
			Matrix += NNParams.HiddenNeurons;

			//Hidden hidden loop
			for(int c = 1; c < NNParams.HiddenLayers; c++)
			{
				//Swap input and output
				Temp = Input;
				Input = Output;
				Output = Temp;

				//Multiply
				cublasSgemv(Handle, CUBLAS_OP_N,
					NNParams.HiddenNeurons, NNParams.HiddenNeurons, &Alpha,
					Matrix, NNParams.HiddenNeurons, Input, 1, &Beta, Output, 1);
				cudaDeviceSynchronize();

				Matrix += NNParams.HiddenNeurons * NNParams.HiddenNeurons;

				//Add biases
				cublasSaxpy(Handle, NNParams.HiddenNeurons,
					&Alpha, Matrix, 1, Output, 1);

				//Activation function
				LeakyReLU <<<(NNParams.HiddenNeurons - 1) / 32 + 1, 32>>> (Output, NNParams.HiddenNeurons);
				cudaDeviceSynchronize();

				Matrix += NNParams.HiddenNeurons;
			}

			//Hidden output multiplication + biases
			//Multiply
			cublasSgemv(Handle, CUBLAS_OP_N,
				NNParams.OutputNeurons, NNParams.HiddenNeurons, &Alpha,
				Matrix, NNParams.OutputNeurons, Input, 1, &Beta, Output, 1);
			cudaDeviceSynchronize();

			Matrix += NNParams.HiddenNeurons * NNParams.OutputNeurons;

			//Add biases
			cublasSaxpy(Handle, NNParams.OutputNeurons,
				&Alpha, Matrix, 1, Output, 1);

			//Activation function
			Sigmoid <<<(NNParams.HiddenNeurons - 1) / 32 + 1, 32>>> (Output, NNParams.OutputNeurons);
			cudaDeviceSynchronize();

			Fitness += (NNParams.OutputFeatures[j] - Output[0]) * (NNParams.OutputFeatures[j] - Output[0]);
		}

		Fitness /= NNParams.NumVectors;
		PSOParams.FitnessArray[Index] = Fitness;

		//Ensure that no memory misalignment and access errors occur
		cublasDestroy(Handle);
		//TODO: free any local memory at the end of the kernel
	}
}

// PSO kernel
__global__
void PSO(NNParameters *NNP, PSOParameters *PSOP)
{
	int Index = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ NNParameters NNParams;
	__shared__ PSOParameters PSOParams;

	if(threadIdx.x == 0)
	{
		NNParams = *NNP;
		PSOParams = *PSOP;
	}

	if(Index < PSOParams.NumParticles)
	{
		//Initialize PBest, LBest and fitness
	    float PersonalBest = PSOParams.PersonalBestFitness[Index];
		float PersonalBestX = INF;
		float LocalBestX = INF;
		int LocalBestIndex = Index;

		//Grid and block for network sized transfers
		dim3 NetworkGrid((NNParams.NetworkSize - 1) / 256 + 1, 1, 1);
		dim3 NetworkBlock(256, 1, 1);

	    //Declare r1, r2
		float R1, R2;

	    //Set left and right neighbours
		int Left = (PSOParams.NumParticles + Index - 1) % PSOParams.NumParticles;
		int Right = (1 + Index) % PSOParams.NumParticles;

	    //Initialize random number generator states
	    // curand_init(Index, Index, 0, &PSOParams.States[Index]);
	    curandState_t LocalState = PSOParams.States[Index];

		//Pointer to weights and biases
		float *WeightsAndBiases = &NNParams.WeightsAndBiases[Index * NNParams.NetworkSize];
		float *PersonalBestWeights = &PSOParams.PersonalBestWeights[Index * NNParams.NetworkSize];

		//Load fitness value in local variable
		float Fitness = PSOParams.FitnessArray[Index];

		int Id = 0;

		//Compare fitness to personal best so far
        if(Fitness < PersonalBest)
        {
			//Copy personal best values
            PersonalBest = Fitness;
			PSOParams.PersonalBestFitness[Index] = Fitness;

            //Copy personal best weights and biases
			//Device to device transfer
			DeviceToDevice <<<NetworkGrid, NetworkBlock>>> (PersonalBestWeights, WeightsAndBiases, NNParams.NetworkSize);
			cudaDeviceSynchronize();
        }
		__syncthreads();
        //Update local best particle index (left or right)
        if(PersonalBest > PSOParams.PersonalBestFitness[Left])
            LocalBestIndex = Left;
        if(PersonalBest > PSOParams.PersonalBestFitness[Right])
            LocalBestIndex = Right;
		__syncthreads();

        //Update weights and biases of each particle
		for (int i = 0; i < NNParams.NetworkSize; i++)
		{
			//Set index at which position needs to be updated
			Id = Index * NNParams.NetworkSize + i;

			//Set local best and personal best X (weights / biases)
			LocalBestX = PSOParams.PersonalBestWeights[LocalBestIndex * NNParams.NetworkSize + i];
			PersonalBestX = PSOParams.PersonalBestWeights[Index * NNParams.NetworkSize + i];

			//Generate random numbers
			R1 = curand_uniform(&LocalState);
			R2 = curand_uniform(&LocalState);

			//Update the velocity
			PSOParams.Velocities[Id] = PSOParams.Chi * (PSOParams.Velocities[Id] +
									PSOParams.C1 * R1 * (PersonalBestX - NNParams.WeightsAndBiases[Id]) +
									PSOParams.C2 * R2 * (LocalBestX - NNParams.WeightsAndBiases[Id]));

			//Ensure velocity values are within range
			// if (PSOParams.Velocities[Id] > PSOParams.VMax)
			// 	PSOParams.Velocities[Id] = PSOParams.VMax;
			// if (PSOParams.Velocities[Id] < -PSOParams.VMax)
			// 	PSOParams.Velocities[Id] = -PSOParams.VMax;

			//An interesting observation made today: not restricting the velocity
			//and instead only the position seems to yield much better results than
			//either restricting only the velocity or both or not restricting both

			__syncthreads();
			//Update the position
			NNParams.WeightsAndBiases[Id] = NNParams.WeightsAndBiases[Id] + PSOParams.Velocities[Id];

			// Ensure position values are within range
			if (NNParams.WeightsAndBiases[Id] > PSOParams.XMax)
			{
				NNParams.WeightsAndBiases[Id] = PSOParams.XMax;
				PSOParams.Velocities[Id] = 0.0f;
			}
			if (NNParams.WeightsAndBiases[Id] < -PSOParams.XMax)
			{
				NNParams.WeightsAndBiases[Id] = -PSOParams.XMax;
				PSOParams.Velocities[Id] = 0.0f;
			}
		}
		PSOParams.States[Index] = LocalState;
	}
}

void NeuralNetwork::CheckKernel()
{
	float *a = new float[12];
	float *b = new float[12];

	for(int i = 0; i < 3; i++)
	{
		for(int j = 0; j < 4; j++)
		{
			a[i * 4 + j] = i * 4 + j;
			cout << a[i * 4 + j] << " ";
		}
		cout << endl;
	}

	float *deva, *devb;
	cudaMalloc((void**)&deva, 12 * sizeof(float));
	cudaMalloc((void**)&devb, 12 * sizeof(float));

	cudaMemcpy(deva, a, 12 * sizeof(float), cudaMemcpyHostToDevice);
	dim3 Grid((4 - 1) / TILE_WIDTH + 1, (3 - 1) / TILE_WIDTH + 1, 1);
	dim3 Block(TILE_WIDTH, TILE_WIDTH, 1);
	Transpose <<<Grid, Block>>> (deva, devb, 3, 4);

	cudaMemcpy(b, devb, 12 * sizeof(float), cudaMemcpyDeviceToHost);

	for(int i = 0; i < 4; i++)
	{
		for(int j = 0; j < 3; j++)
		{
			cout << b[i * 3 + j] << " ";
		}
		cout << endl;
	}
}

//NeuralNetwork::NeuralNetwork()
// Constructor of the NeuralNetwork class
NeuralNetwork::NeuralNetwork(int InputNeurons, int HiddenLayers, int HiddenNeurons, int OutputNeurons, int NumParticles)
{
    //NN hyperparameters
    this->InputNeurons = InputNeurons;
    this->HiddenLayers = HiddenLayers;
    this->HiddenNeurons = HiddenNeurons;
    this->OutputNeurons = OutputNeurons;
    this->NumParticles = NumParticles;
    cout << "HYPERPARAMETERS SET" << endl;

    //Initialize random weights and biases on the GPU
    //Calculate total number of weights and biases for memory allocation
    int NetworkSize = ((InputNeurons + 1) * HiddenNeurons)
                                    + (((HiddenNeurons +1) * HiddenNeurons)
                                        * (HiddenLayers - 1))
                                    + ((HiddenNeurons + 1) * OutputNeurons);
	this->NetworkSize = NetworkSize;

    //Total
    int TotalWeightsAndBiases = NumParticles * NetworkSize;

    cout << "TOTAL SPACE FOR WEIGHTS AND BIASES: " << TotalWeightsAndBiases * 4 / 1024 << "KB" << endl;

    //Allocate device memory for weights and biases
    float *WeightsAndBiases;
    cudaMalloc((void**)&WeightsAndBiases, TotalWeightsAndBiases * sizeof(float));
	cudaCheckError();
    cout << "GPU SPACE ALLOCATED FOR WEIGHTS AND BIASES" << endl;

	//Allocate device memory for weights and biases
    float *PersonalBestWeights;
    cudaMalloc((void**)&PersonalBestWeights, TotalWeightsAndBiases * sizeof(float));
	cudaCheckError();
    cout << "GPU SPACE ALLOCATED FOR PERSONAL BEST WEIGHTS AND BIASES" << endl;

	//Max space to be allocated to intermediate I/O
	int MaxIOLength = 2 * max(InputNeurons, max(HiddenNeurons, OutputNeurons));
	this->MaxIOLength = MaxIOLength;
	float *IntermediateIO;
	cudaMalloc((void**)&IntermediateIO, MaxIOLength * sizeof(float) * this->NumParticles);
	cudaCheckError();
	this->IntermediateIO = IntermediateIO;

    //Allocate device memory for velocities
    float *Velocities;
    cudaMalloc((void**)&Velocities, TotalWeightsAndBiases * sizeof(float));
	cudaCheckError();
    cout << "GPU SPACE ALLOCATED FOR VELOCITIES" << endl;

    //Allocate device memory for fitness values
    float *FitnessArray;
    cudaMalloc((void**)&FitnessArray, NumParticles * sizeof(float));
	cudaCheckError();
    this->FitnessArray = FitnessArray;
    cout << "GPU SPACE ALLOCATED FOR FITNESS VALUES" << endl;

	//InitToVal grid and block
	dim3 InitGrid((this->NumParticles - 1) / 32 + 1, 1, 1);
	dim3 InitBlock(32, 1, 1);

	//Allocate device memory for fitness values
    float *PersonalBestFitness;
    cudaMalloc((void**)&PersonalBestFitness, NumParticles * sizeof(float));
	cudaCheckError();
	InitToVal <<<InitGrid, InitBlock>>> (PersonalBestFitness, this->NumParticles, INF);
	cudaCheckError();
    this->PersonalBestFitness = PersonalBestFitness;
    cout << "GPU SPACE ALLOCATED FOR PERSONAL BEST FITNESS VALUES" << endl;

    //Initialize generator
    curandGenerator_t Gen;
	curandCreateGenerator(&Gen, CURAND_RNG_QUASI_SOBOL32);
	curandSetQuasiRandomGeneratorDimensions(Gen, this->NetworkSize);
	curandSetPseudoRandomGeneratorSeed(Gen, time(NULL));
    cout << "CURAND GENERATOR INITIALIZED" << endl;

    //Dim3 variables for Normalize kernel
    dim3 NormalizeGrid(NetworkSize, 1, 1);
    dim3 NormalizeBlock(NumParticles, 1, 1);

	//Transpose grid and block
	dim3 TransposeGrid((this->NumParticles - 1) / TILE_WIDTH + 1, (this->NetworkSize - 1) / TILE_WIDTH + 1, 1);
	dim3 TransposeBlock(TILE_WIDTH, TILE_WIDTH, 1);

	//Generate weights and biases
    curandGenerateUniform(Gen, WeightsAndBiases, TotalWeightsAndBiases);
    Normalize <<<NormalizeGrid, NormalizeBlock>>> (WeightsAndBiases, TotalWeightsAndBiases, 10.0f);
	cudaCheckError();
	Transpose <<<TransposeGrid, TransposeBlock>>> (WeightsAndBiases, PersonalBestWeights, this->NetworkSize, this->NumParticles);
	cudaCheckError();
    this->WeightsAndBiases = WeightsAndBiases;
    cout << "WEIGHTS AND BIASES INITIALIZED ON GPU" << endl;

	//Copy generated weights and biases to personal best array for initialization
	DeviceToDevice <<<NormalizeGrid, NormalizeBlock>>> (WeightsAndBiases, PersonalBestWeights, TotalWeightsAndBiases);
	this->PersonalBestWeights = PersonalBestWeights;

    //Generate velocities
    curandGenerateUniform(Gen, Velocities, TotalWeightsAndBiases);
    Normalize <<<NormalizeGrid, NormalizeBlock>>> (Velocities, TotalWeightsAndBiases, 1.0f);
	cudaCheckError();
    this->Velocities = Velocities;
    cout << "VELOCITIES INITIALIZED ON GPU" << endl;

    //Allocate space for curand states
    curandState_t *States;
    cudaMalloc((void**)&States, NumParticles * sizeof(curandState_t));
	cudaCheckError();
	InitRNGStates <<<InitGrid, InitBlock>>> (States, this->NumParticles);
	cudaCheckError();
    this->States = States;
    cout << "SPACE ALLOCATED FOR CURAND STATES" << endl;

    //Synchronize all kernel calls upto this point
    cudaDeviceSynchronize();
}

// NeuralNetwork::Load()
// Loads the input feature vectors into an array on the CPU and transfers it to
// the GPU. Method of transferring and thus training (with or without streams)
// will vary depending upon the size of input data.
void NeuralNetwork::Load(const char *FileName)
{
    int Size;
    float *InputFeatures;
    float *OutputFeatures;
    int Width = this->InputNeurons;
    fstream FIn;
    FIn.open(FileName);
    if(!FIn.fail())
    {
        cout << "FILE OPENED" << endl;
        FIn >> Size;
        InputFeatures = new float[Size * Width];
        OutputFeatures = new float[Size];
        cout << "SPACE ALLOCATED" << endl;
        int temp;

        for(int i = 0; i < Size; i++)
        {
            for(int j = 0; j < Width; j++)
            {
                FIn >> temp;
                InputFeatures[i * Width + j] = float(temp);
            }
            FIn >> temp;
            OutputFeatures[i] = float(temp);
        }
    }
    FIn.close();

    cout << "INPUT OUTPUT SPACE REQUIRED: " << Size * 24 / 1024 << "KB" << endl;
	this->NumVectors = Size;

    cout << "INPUT AND OUTPUT LOADED AND FILE CLOSED" << endl;

    //Transfer to GPU (Single cudaMemcpy() for the time being)
    float* DeviceInputFeatures;
    cudaMalloc((void**)&DeviceInputFeatures, Size * Width * sizeof(float));
	cudaCheckError();
    cudaMemcpy(DeviceInputFeatures, InputFeatures, Size * Width * sizeof(float), cudaMemcpyHostToDevice);
	cudaCheckError();
    this->InputFeatures = DeviceInputFeatures;

    float* DeviceOutputFeatures;
    cudaMalloc((void**)&DeviceOutputFeatures, Size * sizeof(float));
	cudaCheckError();
    cudaMemcpy(DeviceOutputFeatures, OutputFeatures, Size * sizeof(float), cudaMemcpyHostToDevice);
	cudaCheckError();
    this->OutputFeatures = DeviceOutputFeatures;

    cout << "INPUT AND OUTPUT TRANSFERRED TO GPU" << endl;
}

// NeuralNetwork::Train()
// Trains the network using PSO and a set number of particles in order to eliminate
// backpropogation.
// Assumes weight matrix to be in column major format.
void NeuralNetwork::Train(int Epochs, const char *WeightsFile, bool Verbose)
{
    dim3 Grid((this->NumParticles - 1) / 32 + 1, 1, 1);
    dim3 Block(32, 1, 1);

	//NN parameters struct
	NNParameters NNParams;
	NNParams.Epochs = Epochs;
	NNParams.InputNeurons = this->InputNeurons;
	NNParams.HiddenLayers = this->HiddenLayers;
	NNParams.HiddenNeurons = this->HiddenNeurons;
	NNParams.OutputNeurons = this->OutputNeurons;
	NNParams.NetworkSize = this->NetworkSize;
	NNParams.MaxIOLength = this->MaxIOLength;
	NNParams.NumVectors = this->NumVectors;
	NNParams.InputFeatures = this->InputFeatures;
	NNParams.IntermediateIO = this->IntermediateIO;
	NNParams.OutputFeatures = this->OutputFeatures;
	NNParams.WeightsAndBiases = this->WeightsAndBiases;

	//PSO parameters struct
	PSOParameters PSOParams;
	PSOParams.NumParticles = this->NumParticles;
	PSOParams.C1 = 2.05f;
	PSOParams.C2 = 2.05f;
	float Psi = PSOParams.C1 + PSOParams.C2;
	float Chi = abs(2.0f / (2.0f - Psi - sqrt(Psi * Psi - 4.0f * Psi)));
	PSOParams.Chi = Chi;
	PSOParams.XMax = 10.0f;
	PSOParams.VMax = 1.0f;
	PSOParams.FitnessArray = this->FitnessArray;
	PSOParams.PersonalBestFitness = this->PersonalBestFitness;
	PSOParams.States = this->States;
	PSOParams.PersonalBestWeights = this->PersonalBestWeights;
	PSOParams.Velocities = this->Velocities;

	NNParameters *D_NNParams;
	PSOParameters *D_PSOParams;

	cudaMalloc((void**)&D_NNParams, sizeof(NNParameters));
	cudaCheckError();
	cudaMalloc((void**)&D_PSOParams, sizeof(PSOParameters));
	cudaCheckError();

	cudaMemcpy(D_NNParams, &NNParams, sizeof(NNParameters), cudaMemcpyHostToDevice);
	cudaCheckError();
	cudaMemcpy(D_PSOParams, &PSOParams, sizeof(PSOParameters), cudaMemcpyHostToDevice);
	cudaCheckError();

	float *Results = new float[this->NumParticles];
	int BestIndex = 0;
	float Best = INF;

    //Train using PSO
	for(int i = 0; i < Epochs; i++)
	{
		cout << "EPOCH (" << i + 1  << " / " << Epochs << ")" << endl;
		FeedForward <<<Grid, Block>>> (D_NNParams, D_PSOParams);
		cudaDeviceSynchronize();
		cudaCheckError();
		PSO <<<Grid, Block>>> (D_NNParams, D_PSOParams);
		cudaDeviceSynchronize();
		cudaCheckError();

		if(Verbose)
		{
			cudaMemcpy(Results, PSOParams.PersonalBestFitness, this->NumParticles * sizeof(float), cudaMemcpyDeviceToHost);
			cudaCheckError();
			BestIndex = 0;
			Best = Results[0];
			cout << "[" << Results[0];
			for(int j = 1; j < this->NumParticles; j++)
			{
				if(Best > Results[j])
				{
					BestIndex = j;
					Best = Results[j];
				}
				cout << ", " << Results[j];
			}
			cout << "]" << endl;
			cout << "BEST PARTICLE: " << BestIndex << endl;
			cout << "BEST FITNESS: " << Best << endl;
		}
	}

	cudaMemcpy(Results, PSOParams.PersonalBestFitness, this->NumParticles * sizeof(float), cudaMemcpyDeviceToHost);
	cudaCheckError();

	BestIndex = 0;
	Best = Results[0];
	for(int i = 1; i < this->NumParticles; i++)
	{
		if(Best > Results[i])
		{
			BestIndex = i;
			Best = Results[i];
		}
	}

	cout << "FINAL BEST PARTICLE: " << BestIndex << endl;
	cout << "FINAL BEST FITNESS: " << Best << endl;

	float *DeviceBestNetwork = &this->PersonalBestWeights[this->NetworkSize * BestIndex];
	float *BestNetwork = new float[this->NetworkSize];
	cudaMemcpy(BestNetwork, DeviceBestNetwork, this->NetworkSize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaCheckError();

	//Dump to file
	fstream FOut;
    FOut.open(WeightsFile, fstream::out);
    if(!FOut.fail())
	{
		FOut << this->InputNeurons << endl;
		FOut << this->HiddenLayers << endl;
		FOut << this->HiddenNeurons << endl;
		FOut << this->OutputNeurons << endl;
		for(int i = 0; i < this->NetworkSize; i++)
		{
			FOut << BestNetwork[i] << endl;
		}
	}
	FOut.close();
}

// NeuralNetwork::Test()
// Tests a set of weights and biases and reports the loss
void NeuralNetwork::Test(const char *TestFile, const char *WeightsFile)
{
	fstream FIn;
	int InputNeurons = 0;
	int HiddenLayers = 0;
	int HiddenNeurons = 0;
	int OutputNeurons = 0;
	int NetworkSize = 0;
	float *Weights;
	FIn.open(WeightsFile, fstream::in);
	if(!FIn.fail())
	{
		FIn >> InputNeurons;
		FIn >> HiddenLayers;
		FIn >> HiddenNeurons;
		FIn >> OutputNeurons;

		NetworkSize = ((InputNeurons + 1) * HiddenNeurons)
                            + (((HiddenNeurons +1) * HiddenNeurons)
                                * (HiddenLayers - 1))
                            + ((HiddenNeurons + 1) * OutputNeurons);

		Weights = new float[NetworkSize];
		for(int i = 0; i < NetworkSize; i++)
			FIn >> Weights[i];
	}
	FIn.close();

	int NumSamples = 0;
	float *InputFeatures;
	float *OutputFeatures;
	FIn.open(TestFile, fstream::in);
	if(!FIn.fail())
	{
		FIn >> NumSamples;
		InputFeatures = new float[NumSamples * InputNeurons];
		OutputFeatures = new float[NumSamples];

		for(int i = 0; i < NumSamples; i++)
		{
			for(int j = 0; j < InputNeurons; j++)
				FIn >> InputFeatures[i * InputNeurons + j];

			FIn >> OutputFeatures[i];
		}
	}
	FIn.close();

	float *InputVectors;
	cudaMalloc((void**)&InputVectors, NumSamples * InputNeurons * sizeof(float));
	cudaCheckError();
	cudaMemcpy(InputVectors, InputFeatures, NumSamples * InputNeurons * sizeof(float), cudaMemcpyHostToDevice);
	cudaCheckError();

	float *WeightsAndBiases;
	cudaMalloc((void**)&WeightsAndBiases, NetworkSize * sizeof(float));
	cudaCheckError();
	cudaMemcpy(WeightsAndBiases, Weights, NetworkSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaCheckError();

	cublasHandle_t Handle;
	cublasCreate(&Handle);

	float Alpha = 1.0f, Beta = 0.0f;
	float Fitness = 0.0f, TempFitness = 0.0f;
	float *Input, *Output, *Matrix, * Temp;

	int MaxIOLength = 2 * max(InputNeurons, max(HiddenNeurons, OutputNeurons));
	float *IntermediateIO;
	cudaMalloc((void**)&IntermediateIO, MaxIOLength * sizeof(float));
	cudaCheckError();

	//Main feed forward work to be done here
	//Calculate fitness, i.e. loss (MSE?)
	for(int j = 0; j < NumSamples; j++)
	{
		//Input hidden multiplication + biases
		Input = &InputVectors[InputNeurons * j];
		Output = IntermediateIO;
		Matrix = WeightsAndBiases;

		cublasSgemv(Handle, CUBLAS_OP_N,
			HiddenNeurons, InputNeurons, &Alpha,
			Matrix, HiddenNeurons, Input, 1, &Beta, Output, 1);
		cudaDeviceSynchronize();

		Matrix += InputNeurons * HiddenNeurons;

		//Add biases
		cublasSaxpy(Handle, HiddenNeurons,
			&Alpha, Matrix, 1, Output, 1);
		cudaDeviceSynchronize();

		//Activation function
		LeakyReLU <<<(HiddenNeurons - 1) / 32 + 1, 32>>> (Output, HiddenNeurons);
		cudaCheckError();
		cudaDeviceSynchronize();

		Input = Output + MaxIOLength / 2;
		Matrix += HiddenNeurons;

		//Hidden hidden loop
		for(int c = 1; c < HiddenLayers; c++)
		{
			//Swap input and output
			Temp = Input;
			Input = Output;
			Output = Temp;

			//Multiply
			cublasSgemv(Handle, CUBLAS_OP_N,
				HiddenNeurons, HiddenNeurons, &Alpha,
				Matrix, HiddenNeurons, Input, 1, &Beta, Output, 1);
			cudaDeviceSynchronize();

			Matrix += HiddenNeurons * HiddenNeurons;

			//Add biases
			cublasSaxpy(Handle, HiddenNeurons,
				&Alpha, Matrix, 1, Output, 1);
			cudaDeviceSynchronize();

			//Activation function
			LeakyReLU <<<(HiddenNeurons - 1) / 32 + 1, 32>>> (Output, HiddenNeurons);
			cudaCheckError();
			cudaDeviceSynchronize();

			Matrix += HiddenNeurons;
		}

		//Hidden output multiplication + biases
		//Multiply
		cublasSgemv(Handle, CUBLAS_OP_N,
			OutputNeurons, HiddenNeurons, &Alpha,
			Matrix, OutputNeurons, Input, 1, &Beta, Output, 1);
		cudaDeviceSynchronize();

		Matrix += HiddenNeurons * OutputNeurons;

		//Add biases
		cublasSaxpy(Handle, OutputNeurons,
			&Alpha, Matrix, 1, Output, 1);
		cudaDeviceSynchronize();

		//Activation function
		Sigmoid <<<(OutputNeurons - 1) / 32 + 1, 32>>> (Output, OutputNeurons);
		cudaCheckError();
		cudaDeviceSynchronize();

		cudaMemcpy(&TempFitness, Output, sizeof(float), cudaMemcpyDeviceToHost);
		Fitness += (OutputFeatures[j] - TempFitness) * (OutputFeatures[j] - TempFitness);
	}

	cublasDestroy(Handle);
	Fitness /= NumSamples;

	cout << "TEST FITNESS: " << Fitness << endl;
}
