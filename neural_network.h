#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdio>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

typedef struct PSOParameters
{
	curandState_t *States = NULL;
	int NumParticles = 0;
	float *FitnessArray = NULL;
	float *PersonalBestFitness = NULL;
	float *PersonalBestWeights = NULL;
	float *Velocities = NULL;
	float C1 = 0.0f;
	float C2 = 0.0f;
	float Chi = 0.0f;
	float XMax = 0.0f;
	float VMax = 0.0f;
} PSOParameters;

typedef struct NNParameters
{
	int Epochs = 0;
	int InputNeurons = 0;
	int HiddenLayers = 0;
	int HiddenNeurons = 0;
	int OutputNeurons = 0;
	int NetworkSize = 0;
	int MaxIOLength = 0;
	int NumVectors = 0;
	float *WeightsAndBiases = NULL;
	float *InputFeatures = NULL;
	float *IntermediateIO = NULL;
	float *OutputFeatures = NULL;
} NNParameters;

class NeuralNetwork
{
private:
    int InputNeurons, HiddenLayers, HiddenNeurons, OutputNeurons;
    int NumParticles, NetworkSize;
    int MaxIOLength, NumVectors;
    float *WeightsAndBiases;
    float *InputFeatures, *OutputFeatures, *IntermediateIO;
    float *Velocities, *FitnessArray, *PersonalBestFitness, *PersonalBestWeights;
    curandState_t *States;

public:
    //Randomly initialize weights and biases for all particles of the swarm
    NeuralNetwork(int, int, int, int, int);

    //Load()
    //Load data from a file into the main memory/GPU memory (as needed)
    //Reshape data if needed (especially separating input features from output labels)
    //Set up streams later on if needed
    void Load(const char *);

    //Train()
    //FeedForward combined with PSO
    //Number of particles taken from constructor
    void Train(int, const char *, bool);


    //Test()
    //Use the best set of weights and biases amongst all particles
    void Test(const char *, const char *);

    //Dump()
    //Dump the best set of weights and biases to a file
    void CheckKernel();
};

#endif
