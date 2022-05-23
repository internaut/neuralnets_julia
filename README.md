# Neural Network from Scratch in Julia

Adapted from the original Python/PyTorch implemenation at https://sirupsen.com/napkin/neural-net.

May 2022 / Markus Konrad <post@mkonrad.net>

The main file `neuralnets.jl` shows an example implementation of a single layer neural network with Julia. It uses the [`AutoGrad` package](https://juliapackages.com/p/autograd) for calculating the loss function gradients for stochastic gradient descend (SGD). For a better understanding of `AutoGrad` two more files are added: `autograd_tanh.jl` shows how to get the derivate of a user defined function and `autograd_lm.jl` shows how to apply SGD for a simple linear model with two parameters.

You should run the code line-by-line with a Julia REPL in order to understand it.
