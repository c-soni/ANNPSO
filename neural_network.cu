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

// Normalizes a vector of values in [0, 1] to [-MaxValue, MaxValue]
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

// Initialize to value
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

// Sigmoid activiation function
__global__
void Sigmoid(float *Input, int Size)
{
    int Index = blockDim.x * blockIdx.x + threadIdx.x;
    if(Index < Size)
        Input[Index] = (1 / (1 + __expf(-Input[Index])));
}

// SquaredError loss function
__global__
void SquaredError(float *Predicted, float *Actual, float *Fitness, int Size)
{
    int Index = blockDim.x * blockIdx.x + threadIdx.x;
    if(Index < Size)
        Fitness[Index] += (Predicted[Index] - (*Actual)) * (Predicted[Index] - (*Actual));
}

// Mean function
__global__
void Mean(float *Input, int NumElements, int Size)
{
    int Index = blockDim.x * blockIdx.x + threadIdx.x;
    if(Index < Size)
        Input[Index] /= NumElements;
}

// Kernel which actually trains the data.
// __global__
// void FeedForward(NNParameters *NNP, PSOParameters *PSOP)
// {
//     int Index = blockDim.x * blockIdx.x + threadIdx.x;
//     __shared__ NNParameters NNParams;
//     __shared__ PSOParameters PSOParams;

//     if(threadIdx.x == 0)
//     {
//         NNParams = *NNP;
//         PSOParams = *PSOP;
//     }

//     if(Index < PSOParams.NumParticles)
//     {
//         //Pointer to weights and biases
//         float *WeightsAndBiases = &NNParams.WeightsAndBiases[Index * NNParams.NetworkSize];

//         //Input, output, matrix and temporary pointers
//         float *Input;
//         float *Output;
//         float *Matrix;
//         float *Temp;

//         //Fitness value
//         float Fitness = 0.0f;

//         //cuBLAS handle initialization
//         cublasHandle_t Handle;
//         cublasCreate(&Handle);

//         //Alpha and beta values
//         float Alpha = 1.0f;
//         float Beta = 0.0f;

//         Fitness = 0.0f;

//         //Main feed forward work to be done here
//         //Calculate fitness, i.e. loss (MSE?)
//         for(int j = 0; j < NNParams.NumVectors; j++)
//         {
//             //Input hidden multiplication + biases
//             Input = &(NNParams.InputFeatures[NNParams.InputNeurons * j]);
//             Output = &(NNParams.IntermediateIO[NNParams.MaxIOLength * Index]);
//             Matrix = &(NNParams.WeightsAndBiases[NNParams.NetworkSize * Index]);

//             cublasSgemv(Handle, CUBLAS_OP_N,
//                 NNParams.HiddenNeurons, NNParams.InputNeurons, &Alpha,
//                 Matrix, NNParams.HiddenNeurons, Input, 1, &Beta, Output, 1);
//             cudaDeviceSynchronize();

//             Matrix += NNParams.InputNeurons * NNParams.HiddenNeurons;

//             //Add biases
//             cublasSaxpy(Handle, NNParams.HiddenNeurons,
//                 &Alpha, Matrix, 1, Output, 1);

//             //Activation function
//             LeakyReLU <<<(NNParams.HiddenNeurons - 1) / 32 + 1, 32>>> (Output, NNParams.HiddenNeurons);
//             cudaDeviceSynchronize();

//             Input = Output + NNParams.MaxIOLength / 2;
//             Matrix += NNParams.HiddenNeurons;

//             //Hidden hidden loop
//             for(int c = 1; c < NNParams.HiddenLayers; c++)
//             {
//                 //Swap input and output
//                 Temp = Input;
//                 Input = Output;
//                 Output = Temp;

//                 //Multiply
//                 cublasSgemv(Handle, CUBLAS_OP_N,
//                     NNParams.HiddenNeurons, NNParams.HiddenNeurons, &Alpha,
//                     Matrix, NNParams.HiddenNeurons, Input, 1, &Beta, Output, 1);
//                 cudaDeviceSynchronize();

//                 Matrix += NNParams.HiddenNeurons * NNParams.HiddenNeurons;

//                 //Add biases
//                 cublasSaxpy(Handle, NNParams.HiddenNeurons,
//                     &Alpha, Matrix, 1, Output, 1);

//                 //Activation function
//                 LeakyReLU <<<(NNParams.HiddenNeurons - 1) / 32 + 1, 32>>> (Output, NNParams.HiddenNeurons);
//                 cudaDeviceSynchronize();

//                 Matrix += NNParams.HiddenNeurons;
//             }

//             //Hidden output multiplication + biases
//             //Multiply
//             cublasSgemv(Handle, CUBLAS_OP_N,
//                 NNParams.OutputNeurons, NNParams.HiddenNeurons, &Alpha,
//                 Matrix, NNParams.OutputNeurons, Input, 1, &Beta, Output, 1);
//             cudaDeviceSynchronize();

//             Matrix += NNParams.HiddenNeurons * NNParams.OutputNeurons;

//             //Add biases
//             cublasSaxpy(Handle, NNParams.OutputNeurons,
//                 &Alpha, Matrix, 1, Output, 1);

//             //Activation function
//             Sigmoid <<<(NNParams.OutputNeurons - 1) / 32 + 1, 32>>> (Output, NNParams.OutputNeurons);
//             cudaDeviceSynchronize();

//             Fitness += (NNParams.OutputFeatures[j] - Output[0]) * (NNParams.OutputFeatures[j] - Output[0]);
//         }

//         Fitness /= NNParams.NumVectors;
//         PSOParams.FitnessArray[Index] = Fitness;

//         //Ensure that no memory misalignment and access errors occur
//         cublasDestroy(Handle);
//         //TODO: free any local memory at the end of the kernel
//     }
// }

// FeedForward function on CPU w/o cuBLAS Device API
void NeuralNetwork::FeedForward(NNParameters &NNParams, PSOParameters &PSOParams)
{
    //cuBLAS handle initialization
    cublasHandle_t Handle;
    cublasCreate(&Handle);

    //Alpha and beta values
    float Alpha = 1.0f;
    float Beta = 0.0f;

    //Input, output, matrix and temporary pointers
    float *Input;
    float *Output;
    float *Matrix;
    float *Temp;

    for(int j = 0; j < NNParams.NumVectors; j++)
    {
        for(int i = 0; i < PSOParams.NumParticles; i++)
        {
            Input = NNParams.InputFeatures + (NNParams.InputNeurons * j);
            Output = NNParams.IntermediateIO + (NNParams.MaxIOLength * i);
            Matrix = NNParams.WeightsAndBiases + (NNParams.NetworkSize * i);

            //Main feed forward work to be done here
            //Calculate fitness, i.e. loss (MSE?)

            //Input hidden multiplication + biases
            cublasSgemv(Handle, CUBLAS_OP_N,
                NNParams.HiddenNeurons, NNParams.InputNeurons, &Alpha,
                Matrix, NNParams.HiddenNeurons, Input, 1, &Beta, Output, 1);

            Matrix += NNParams.InputNeurons * NNParams.HiddenNeurons;

            //Add biases
            cublasSaxpy(Handle, NNParams.HiddenNeurons,
                &Alpha, Matrix, 1, Output, 1);

            //Activation function
            LeakyReLU <<<(NNParams.HiddenNeurons - 1) / 32 + 1, 32>>> (Output, NNParams.HiddenNeurons);

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

                Matrix += NNParams.HiddenNeurons * NNParams.HiddenNeurons;

                //Add biases
                cublasSaxpy(Handle, NNParams.HiddenNeurons,
                    &Alpha, Matrix, 1, Output, 1);

                //Activation function
                LeakyReLU <<<(NNParams.HiddenNeurons - 1) / 32 + 1, 32>>> (Output, NNParams.HiddenNeurons);

                Matrix += NNParams.HiddenNeurons;
            }

            //Hidden output multiplication + biases
            //Multiply
            cublasSgemv(Handle, CUBLAS_OP_N,
                NNParams.OutputNeurons, NNParams.HiddenNeurons, &Alpha,
                Matrix, NNParams.OutputNeurons, Input, 1, &Beta, Output, 1);

            Matrix += NNParams.HiddenNeurons * NNParams.OutputNeurons;

            //Add biases
            cublasSaxpy(Handle, NNParams.OutputNeurons,
                &Alpha, Matrix, 1, Output, 1);

            //Activation function
            Sigmoid <<<(NNParams.OutputNeurons - 1) / 32 + 1, 32>>> (Output, NNParams.OutputNeurons);
            // MeanSquaredError <<<(NNParams.OutputNeurons - 1) / 32 + 1, 32>>> (PSOParams.FitnessArray, Output, NNParams.OutputNeurons);
        }

        //Calculate fitness
        float *OutputFeaturesPointer = NNParams.OutputFeatures + j;
        SquaredError <<<(PSOParams.NumParticles - 1) / 32 + 1, 32>>> (Output, OutputFeaturesPointer, PSOParams.FitnessArray, PSOParams.NumParticles);
    }

    // Calculate mean fitness
    Mean <<<(PSOParams.NumParticles - 1) / 32 + 1, 32>>> (PSOParams.FitnessArray, NNParams.NumVectors, PSOParams.NumParticles);

    //Ensure that no memory misalignment and access errors occur
    cublasDestroy(Handle);
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
            //     PSOParams.Velocities[Id] = PSOParams.VMax;
            // if (PSOParams.Velocities[Id] < -PSOParams.VMax)
            //     PSOParams.Velocities[Id] = -PSOParams.VMax;

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
            std::cout << a[i * 4 + j] << " ";
        }
        std::cout << std::endl;
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
            std::cout << b[i * 3 + j] << " ";
        }
        std::cout << std::endl;
    }
}

//NeuralNetwork::NeuralNetwork()
// Constructor of the NeuralNetwork class
NeuralNetwork::NeuralNetwork(int InputNeurons, int HiddenLayers, int HiddenNeurons, int OutputNeurons, int NumParticles)
{
    //NN hyperparameters
    this->NNParams.InputNeurons = InputNeurons;
    this->NNParams.HiddenLayers = HiddenLayers;
    this->NNParams.HiddenNeurons = HiddenNeurons;
    this->NNParams.OutputNeurons = OutputNeurons;
    this->PSOParams.NumParticles = NumParticles;
    std::cout << "HYPERPARAMETERS SET" << std::endl;

    //Initialize random weights and biases on the GPU
    //Calculate total number of weights and biases for memory allocation
    int NetworkSize = ((InputNeurons + 1) * HiddenNeurons)
                                    + (((HiddenNeurons +1) * HiddenNeurons)
                                        * (HiddenLayers - 1))
                                    + ((HiddenNeurons + 1) * OutputNeurons);
    this->NNParams.NetworkSize = NetworkSize;

    //Total
    int TotalWeightsAndBiases = NumParticles * NetworkSize;

    std::cout << "TOTAL SPACE FOR WEIGHTS AND BIASES: " << TotalWeightsAndBiases * 4 / 1024 << "KB" << std::endl;

    //Allocate device memory for weights and biases
    float *WeightsAndBiases;
    cudaMalloc((void**)&WeightsAndBiases, TotalWeightsAndBiases * sizeof(float));
    cudaCheckError();
    std::cout << "GPU SPACE ALLOCATED FOR WEIGHTS AND BIASES" << std::endl;

    //Allocate device memory for weights and biases
    float *PersonalBestWeights;
    cudaMalloc((void**)&PersonalBestWeights, TotalWeightsAndBiases * sizeof(float));
    cudaCheckError();
    std::cout << "GPU SPACE ALLOCATED FOR PERSONAL BEST WEIGHTS AND BIASES" << std::endl;

    //Max space to be allocated to intermediate I/O
    int MaxIOLength = 2 * max(InputNeurons, max(HiddenNeurons, OutputNeurons));
    this->NNParams.MaxIOLength = MaxIOLength;
    float *IntermediateIO;
    cudaMalloc((void**)&IntermediateIO, MaxIOLength * sizeof(float) * this->PSOParams.NumParticles);
    cudaCheckError();
    this->NNParams.IntermediateIO = IntermediateIO;

    //Allocate device memory for velocities
    float *Velocities;
    cudaMalloc((void**)&Velocities, TotalWeightsAndBiases * sizeof(float));
    cudaCheckError();
    std::cout << "GPU SPACE ALLOCATED FOR VELOCITIES" << std::endl;

    //InitToVal grid and block
    dim3 InitGrid((this->PSOParams.NumParticles - 1) / 32 + 1, 1, 1);
    dim3 InitBlock(32, 1, 1);

    //Allocate device memory for fitness values
    float *FitnessArray;
    cudaMalloc((void**)&FitnessArray, NumParticles * sizeof(float));
    cudaCheckError();
    InitToVal <<<InitGrid, InitBlock>>> (FitnessArray, this->PSOParams.NumParticles, 0.0f);
    cudaCheckError();
    this->PSOParams.FitnessArray = FitnessArray;
    std::cout << "GPU SPACE ALLOCATED FOR FITNESS VALUES" << std::endl;

    //Allocate device memory for fitness values
    float *PersonalBestFitness;
    cudaMalloc((void**)&PersonalBestFitness, NumParticles * sizeof(float));
    cudaCheckError();
    InitToVal <<<InitGrid, InitBlock>>> (PersonalBestFitness, this->PSOParams.NumParticles, INF);
    cudaCheckError();
    this->PSOParams.PersonalBestFitness = PersonalBestFitness;
    std::cout << "GPU SPACE ALLOCATED FOR PERSONAL BEST FITNESS VALUES" << std::endl;

    //Initialize generator
    curandGenerator_t Gen;
    curandCreateGenerator(&Gen, CURAND_RNG_QUASI_SOBOL32);
    curandSetQuasiRandomGeneratorDimensions(Gen, this->NNParams.NetworkSize);
    curandSetPseudoRandomGeneratorSeed(Gen, time(NULL));
    std::cout << "CURAND GENERATOR INITIALIZED" << std::endl;

    //Dim3 variables for Normalize kernel
    dim3 NormalizeGrid(NetworkSize, 1, 1);
    dim3 NormalizeBlock(NumParticles, 1, 1);

    //Transpose grid and block
    dim3 TransposeGrid((this->PSOParams.NumParticles - 1) / TILE_WIDTH + 1, (this->NNParams.NetworkSize - 1) / TILE_WIDTH + 1, 1);
    dim3 TransposeBlock(TILE_WIDTH, TILE_WIDTH, 1);

    //Generate weights and biases
    curandGenerateUniform(Gen, WeightsAndBiases, TotalWeightsAndBiases);
    Normalize <<<NormalizeGrid, NormalizeBlock>>> (WeightsAndBiases, TotalWeightsAndBiases, 10.0f);
    cudaCheckError();
    Transpose <<<TransposeGrid, TransposeBlock>>> (WeightsAndBiases, PersonalBestWeights, this->NNParams.NetworkSize, this->PSOParams.NumParticles);
    cudaCheckError();
    this->NNParams.WeightsAndBiases = WeightsAndBiases;
    std::cout << "WEIGHTS AND BIASES INITIALIZED ON GPU" << std::endl;

    //Copy generated weights and biases to personal best array for initialization
    DeviceToDevice <<<NormalizeGrid, NormalizeBlock>>> (WeightsAndBiases, PersonalBestWeights, TotalWeightsAndBiases);
    this->PSOParams.PersonalBestWeights = PersonalBestWeights;

    //Generate velocities
    curandGenerateUniform(Gen, Velocities, TotalWeightsAndBiases);
    Normalize <<<NormalizeGrid, NormalizeBlock>>> (Velocities, TotalWeightsAndBiases, 1.0f);
    cudaCheckError();
    this->PSOParams.Velocities = Velocities;
    std::cout << "VELOCITIES INITIALIZED ON GPU" << std::endl;

    //Allocate space for curand states
    curandState_t *States;
    cudaMalloc((void**)&States, NumParticles * sizeof(curandState_t));
    cudaCheckError();
    InitRNGStates <<<InitGrid, InitBlock>>> (States, this->PSOParams.NumParticles);
    cudaCheckError();
    this->PSOParams.States = States;
    std::cout << "SPACE ALLOCATED FOR CURAND STATES" << std::endl;

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
    int InputWidth = this->NNParams.InputNeurons;
    int OutputWidth = this->NNParams.OutputNeurons;
    std::fstream FIn;
    FIn.open(FileName);
    if(!FIn.fail())
    {
        std::cout << "FILE OPENED" << std::endl;
        FIn >> Size;
        InputFeatures = new float[Size * InputWidth];
        OutputFeatures = new float[Size];
        std::cout << "SPACE ALLOCATED" << std::endl;
        int temp;

        for(int i = 0; i < Size; i++)
        {
            for(int j = 0; j < InputWidth; j++)
            {
                FIn >> temp;
                InputFeatures[i * InputWidth + j] = float(temp);
            }
            for(int j = 0; j < OutputWidth; j++)
            {
                FIn >> temp;
                OutputFeatures[i * OutputWidth + j] = float(temp);
            }
        }
    }
    FIn.close();

    std::cout << "INPUT OUTPUT SPACE REQUIRED: " << Size * 24 / 1024 << "KB" << std::endl;
    this->NNParams.NumVectors = Size;

    std::cout << "INPUT AND OUTPUT LOADED AND FILE CLOSED" << std::endl;

    //Transfer to GPU (Single cudaMemcpy() for the time being)
    float* DeviceInputFeatures;
    cudaMalloc((void**)&DeviceInputFeatures, Size * InputWidth * sizeof(float));
    cudaCheckError();
    cudaMemcpy(DeviceInputFeatures, InputFeatures, Size * InputWidth * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();
    this->NNParams.InputFeatures = DeviceInputFeatures;

    float* DeviceOutputFeatures;
    cudaMalloc((void**)&DeviceOutputFeatures, Size * OutputWidth * sizeof(float));
    cudaCheckError();
    cudaMemcpy(DeviceOutputFeatures, OutputFeatures, Size * OutputWidth * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();
    this->NNParams.OutputFeatures = DeviceOutputFeatures;

    std::cout << "INPUT AND OUTPUT TRANSFERRED TO GPU" << std::endl;
}

// NeuralNetwork::Train()
// Trains the network using PSO and a set number of particles in order to eliminate
// backpropogation.
// Assumes weight matrix to be in column major format.
void NeuralNetwork::Train(int Epochs, const char *WeightsFile, bool Verbose)
{
    dim3 Grid((this->PSOParams.NumParticles - 1) / 32 + 1, 1, 1);
    dim3 Block(32, 1, 1);

    //NN parameters struct
    NNParameters NNParams;
    NNParams.Epochs = Epochs;
    NNParams.InputNeurons = this->NNParams.InputNeurons;
    NNParams.HiddenLayers = this->NNParams.HiddenLayers;
    NNParams.HiddenNeurons = this->NNParams.HiddenNeurons;
    NNParams.OutputNeurons = this->NNParams.OutputNeurons;
    NNParams.NetworkSize = this->NNParams.NetworkSize;
    NNParams.MaxIOLength = this->NNParams.MaxIOLength;
    NNParams.NumVectors = this->NNParams.NumVectors;
    NNParams.InputFeatures = this->NNParams.InputFeatures;
    NNParams.IntermediateIO = this->NNParams.IntermediateIO;
    NNParams.OutputFeatures = this->NNParams.OutputFeatures;
    NNParams.WeightsAndBiases = this->NNParams.WeightsAndBiases;

    //PSO parameters struct
    PSOParameters PSOParams;
    PSOParams.NumParticles = this->PSOParams.NumParticles;
    PSOParams.C1 = 2.05f;
    PSOParams.C2 = 2.05f;
    float Psi = PSOParams.C1 + PSOParams.C2;
    float Chi = abs(2.0f / (2.0f - Psi - sqrt(Psi * Psi - 4.0f * Psi)));
    PSOParams.Chi = Chi;
    PSOParams.XMax = 10.0f;
    PSOParams.VMax = 1.0f;
    PSOParams.FitnessArray = this->PSOParams.FitnessArray;
    PSOParams.PersonalBestFitness = this->PSOParams.PersonalBestFitness;
    PSOParams.States = this->PSOParams.States;
    PSOParams.PersonalBestWeights = this->PSOParams.PersonalBestWeights;
    PSOParams.Velocities = this->PSOParams.Velocities;

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

    float *Results = new float[this->PSOParams.NumParticles];
    int BestIndex = 0;
    float Best = INF;

    //Train using PSO
    for(int i = 0; i < Epochs; i++)
    {
        std::cout << "EPOCH (" << i + 1  << " / " << Epochs << ")" << std::endl;
        // FeedForward <<<Grid, Block>>> (D_NNParams, D_PSOParams);
        FeedForward(NNParams, PSOParams);
        cudaDeviceSynchronize();
        cudaCheckError();
        PSO <<<Grid, Block>>> (D_NNParams, D_PSOParams);
        cudaDeviceSynchronize();
        cudaCheckError();

        if(Verbose)
        {
            cudaMemcpy(Results, PSOParams.PersonalBestFitness, this->PSOParams.NumParticles * sizeof(float), cudaMemcpyDeviceToHost);
            cudaCheckError();
            BestIndex = 0;
            Best = Results[0];
            std::cout << "[" << Results[0];
            for(int j = 1; j < this->PSOParams.NumParticles; j++)
            {
                if(Best > Results[j])
                {
                    BestIndex = j;
                    Best = Results[j];
                }
                std::cout << ", " << Results[j];
            }
            std::cout << "]" << std::endl;
            std::cout << "BEST PARTICLE: " << BestIndex << std::endl;
            std::cout << "BEST FITNESS: " << Best << std::endl;
        }
    }

    cudaMemcpy(Results, PSOParams.PersonalBestFitness, this->PSOParams.NumParticles * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();

    BestIndex = 0;
    Best = Results[0];
    for(int i = 1; i < this->PSOParams.NumParticles; i++)
    {
        if(Best > Results[i])
        {
            BestIndex = i;
            Best = Results[i];
        }
    }

    std::cout << "FINAL BEST PARTICLE: " << BestIndex << std::endl;
    std::cout << "FINAL BEST FITNESS: " << Best << std::endl;

    float *DeviceBestNetwork = &this->PSOParams.PersonalBestWeights[this->NNParams.NetworkSize * BestIndex];
    float *BestNetwork = new float[this->NNParams.NetworkSize];
    cudaMemcpy(BestNetwork, DeviceBestNetwork, this->NNParams.NetworkSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();

    //Dump to file
    std::fstream FOut;
    FOut.open(WeightsFile, std::fstream::out);
    if(!FOut.fail())
    {
        FOut << this->NNParams.InputNeurons << std::endl;
        FOut << this->NNParams.HiddenLayers << std::endl;
        FOut << this->NNParams.HiddenNeurons << std::endl;
        FOut << this->NNParams.OutputNeurons << std::endl;
        for(int i = 0; i < this->NNParams.NetworkSize; i++)
        {
            FOut << BestNetwork[i] << std::endl;
        }
    }
    FOut.close();
}

// NeuralNetwork::Test()
// Tests a set of weights and biases and reports the loss
void NeuralNetwork::Test(const char *TestFile, const char *WeightsFile)
{
    std::fstream FIn;
    int InputNeurons = 0;
    int HiddenLayers = 0;
    int HiddenNeurons = 0;
    int OutputNeurons = 0;
    int NetworkSize = 0;
    float *Weights;
    FIn.open(WeightsFile, std::fstream::in);
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
    FIn.open(TestFile, std::fstream::in);
    if(!FIn.fail())
    {
        FIn >> NumSamples;
        InputFeatures = new float[NumSamples * InputNeurons];
        OutputFeatures = new float[NumSamples * OutputNeurons];

        for(int i = 0; i < NumSamples; i++)
        {
            for(int j = 0; j < InputNeurons; j++)
                FIn >> InputFeatures[i * InputNeurons + j];

            for(int j = 0; j < OutputNeurons; j++)
                FIn >> OutputFeatures[i * OutputNeurons + j];
        }
    }
    FIn.close();

    float *InputVectors;
    cudaMalloc((void**)&InputVectors, NumSamples * InputNeurons * sizeof(float));
    cudaCheckError();
    cudaMemcpy(InputVectors, InputFeatures, NumSamples * InputNeurons * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();

    float *WeightsAndBiases;
    cudaMalloc((void**)&WeightsAndBiases, NetworkSize * OutputNeurons * sizeof(float));
    cudaCheckError();
    cudaMemcpy(WeightsAndBiases, Weights, NetworkSize * OutputNeurons * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();

    cublasHandle_t Handle;
    cublasCreate(&Handle);

    float Alpha = 1.0f, Beta = 0.0f;
    float Fitness = 0.0f, TempFitness = 0.0f;
    float *Input, *Output, *Matrix, *Temp;

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

        Matrix += InputNeurons * HiddenNeurons;

        //Add biases
        cublasSaxpy(Handle, HiddenNeurons,
            &Alpha, Matrix, 1, Output, 1);

        //Activation function
        LeakyReLU <<<(HiddenNeurons - 1) / 32 + 1, 32>>> (Output, HiddenNeurons);
        cudaCheckError();

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

            Matrix += HiddenNeurons * HiddenNeurons;

            //Add biases
            cublasSaxpy(Handle, HiddenNeurons,
                &Alpha, Matrix, 1, Output, 1);

            //Activation function
            LeakyReLU <<<(HiddenNeurons - 1) / 32 + 1, 32>>> (Output, HiddenNeurons);
            cudaCheckError();

            Matrix += HiddenNeurons;
        }

        //Hidden output multiplication + biases
        //Multiply
        cublasSgemv(Handle, CUBLAS_OP_N,
            OutputNeurons, HiddenNeurons, &Alpha,
            Matrix, OutputNeurons, Input, 1, &Beta, Output, 1);

        Matrix += HiddenNeurons * OutputNeurons;

        //Add biases
        cublasSaxpy(Handle, OutputNeurons,
            &Alpha, Matrix, 1, Output, 1);

        //Activation function
        Sigmoid <<<(OutputNeurons - 1) / 32 + 1, 32>>> (Output, OutputNeurons);
        cudaCheckError();

        cudaMemcpy(&TempFitness, Output, OutputNeurons * sizeof(float), cudaMemcpyDeviceToHost);
        Fitness += (OutputFeatures[j] - TempFitness) * (OutputFeatures[j] - TempFitness);
    }

    cublasDestroy(Handle);
    Fitness /= NumSamples;

    std::cout << "TEST FITNESS: " << Fitness << std::endl;
}
