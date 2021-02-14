# ANNPSO

### Repo for the final year college project.

### Proposed idea: Training a neural network with the help of particle swarm optimization (PSO) instead of the traditional backprop and gradient descent.

Current issues with backprop:

1. *Convergence:* Having a single initialization point for weights and biases often leads to a suboptimal convergence at a local minimum. Another major issue is the vanishing gradient problem, i.e., plateau regions in the function topology where the slope is zero.

2. *Derivatives:* Calculating them requires a differentiable non-linear activation function. Thus backprop is inherently ~2x slower than feedforward.

This where PSO comes in: a combinatorial optimization method which:

1. Ensures a good convergence point if done right.

2. Does not require a differentiable "fitness" function.

#### Purely PSO-based:

Currently what this repo is all about. Completely eliminating backprop from the equation and relying solely on PSO.

Results: It has been established that a network is certainly trainable by this method. We over-fit a basic XOR network (5-3-3-3-1) and observed the losses. Using 32 particles, it took roughly 2k iterations for the training loss to drop to 0.06 (~94% training accuracy). Largely over-fit, but trainable nonetheless.

#### A best-of-both-worlds method: [Hybrid PSO with backprop, written in TensorFlow](https://github.com/munagekar/nnpso)

Training using both PSO and backprop for a better rate of convergence. Current results are far better than a purely PSO based approach, but the method itself is a major memory-hog.

##### Issues on this repo serve more as a to-do list than as actual issues.

###### Any suggestions and feedback are welcome. Feel free to raise an issue.
