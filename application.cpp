#include "neural_network.h"

int main()
{
    std::cout << "CONSTRUCTOR INVOKED" << std::endl;
    NeuralNetwork NN(5, 3, 3, 1, 32);

    std::cout << "LOAD INVOKED" << std::endl;
    NN.Load("train.txt");

    std::cout << "TRAIN INVOKED" << std::endl;
    NN.Train(1000, "weights.txt", true);

    std::cout << "TEST INVOKED" << std::endl;
    NN.Test("test.txt", "weights.txt");
}
