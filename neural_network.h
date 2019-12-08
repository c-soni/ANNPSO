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

typedef struct PSOParameters
{
    curandState_t *States;        // Store the state of the GPU random number generator per thread.
    int NumParticles;             // Number of particles in the swarm.
    float *FitnessArray;          // Array for storing the fitness value of all particles.
    float *PersonalBestFitness;   // PBest values of fitness for all particles.
    float *PersonalBestWeights;   // PBest values of weights for all particles.
    float *Velocities;            // Velocities of all particles.
    float C1;                     // Constant value c1.
    float C2;                     // Constant value c2.
    float Chi;                    // Constant value chi.
    float XMax;                   // Upper limit of the sample space, limits position values to [-XMax, XMax].
    float VMax;                   // Upper limit to the velocity, limits velocity values to [-VMax, VMax].
} PSOParameters;

typedef struct NNParameters
{
    int Epochs;                   // Number of epochs to be run.
    int InputNeurons;             // Number of input neurons in the input layer.
    int HiddenLayers;             // Number of hidden neurons per hidden layer.
    int HiddenNeurons;            // Number of hidden layers in the network.
    int OutputNeurons;            // Number of output neurons in the output layer.
    int NetworkSize;              // Total number of neurons in the neural network.
    int MaxIOLength;              // Size of the intermediate IO buffer (number of elements, not bytes).
    int NumVectors;               // Number of samples (feature vectors) in the training set.
    float *WeightsAndBiases;      // Storage buffer for the training weights and biases of all networks.
    float *InputFeatures;         // Storage buffer for training data samples.
    float *IntermediateIO;        // Temporary storage of intermediate IO.
    float *OutputFeatures;        // Storage buffer for training labels.
} NNParameters;

class NeuralNetwork
{
private:
    PSOParameters PSOParams;
    NNParameters NNParams;

public:
    //Randomly initialize weights and biases for all particles of the swarm
    NeuralNetwork(int InputNeurons, int HiddenLayers, int HiddenNeurons, int OutputNeurons, int NumParticles);

    //Load()
    //Load data from a file into the main memory/GPU memory (as needed)
    //Reshape data if needed (especially separating input features from output labels)
    //Set up streams later on if needed
    void Load(const char *FileName);

    //Train()
    //FeedForward combined with PSO
    //Number of particles taken from constructor
    void Train(int Epochs, const char *WeightsFile, bool Verbose);


    //Test()
    //Use the best set of weights and biases amongst all particles
    void Test(const char *TestFile, const char *WeightsFile);

    //Dump()
    //Dump the best set of weights and biases to a file
    void CheckKernel();
};

#endif
