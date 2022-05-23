# Neural Network from Scratch in Julia
# adapted from https://sirupsen.com/napkin/neural-net
#
# May 2022
#
# author: Markus Konrad <post@mkonrad.net>
#

using LinearAlgebra  # for dot
using AutoGrad


# determine the "brightness" of a 2x2 pixels image using a simple "neural network"
#
# this is basically the mean of the input pixels and can be easily implemented by hand (dividing all values by 4 and
# summing) but we want a neural net to find these "weights" (1/4) for each input pixel
# a neural network can approximate any function so it can as well approximate the mean

# input layer determines the input data; here a flattened 2x2 image of pixel intensities in range [0, 1]
input_layer = [0.2, 0.5, 0.4, 0.7]

# the hidden layer has the weights of our computation; here initialized with some random values
hidden_layer = [0.98, 0.4, 0.86, -0.08]

# we calculate the sole output value by mult. each input pixel with its weight from the hidden layer
output_neuron = 0
for (index, input_neuron) in enumerate(input_layer)
    output_neuron += input_neuron * hidden_layer[index]
end

# check output
output_neuron

# this is equiv. to a dot product of two vectors, so we can as well use LinearAlgebra::dot
output_neuron = dot(input_layer, hidden_layer)

# is the same:
output_neuron

# this would be correct for determining the overall brightness of an image:
hidden_layer = fill(0.25, 4)
output_neuron = dot(input_layer, hidden_layer)

# is the mean:
output_neuron

# but we should teach our neural net to find these weights itself

# for this, we first generate some training data: 1000 input images and their overall brightness as mean of their pixel 
# values

mean = x -> sum(x) / length(x)   # defines a mean function

N_TRAINING_SAMPLES = 1000

# preallocate data
#train_data = Vector{Vector{Float64}}(undef, N_TRAINING_SAMPLES)   # this is probably not the right prealloc.
train_data = Matrix{Float64}(undef, (N_TRAINING_SAMPLES, 4))
train_output = Vector{Float64}(undef, N_TRAINING_SAMPLES)

for i in 1:N_TRAINING_SAMPLES
    d = rand(Float64, 4)   # random vector with values in [0, 1)
    train_data[i, :] = d
    train_output[i] = mean(d)
end

# we define a loss function that measures the quality of the current model; we use the mean squared error (MSE)

# x are the actual values, x_hat the predicted values
# note the dots before "-" and "^" operations; they denote that the elementwise operations (the inputs are vectors)
# should be performed
mean_squared_error = (x_hat, x) -> mean((x .- x_hat).^2)

# we define a model; this is simply the weighted sum of the input rows
model = (input_data, weights) -> input_data * weights      # matrix multiplication; equiv. to the following:
#model = (input_data, weights) -> [dot(input_data[i, :], weights) for i in 1:size(input_data)[1]]

# no we define the training function; it takes training data, known training outcomes, the current layer weights and
# a learning rate for the stochastic gradient descend (SGD);
# the function modifies "weights" in-place and returns the current loss
function train(data, outcomes, weights, learningrate=0.1)
    # we define our model as partial function in terms of its parameters, i.e. the weights; the data is "fixed";
    # this is very important so that we later can get the gradients of the loss function w.r.t. to the weights
    model_train = weights -> model(data, weights)

    # we define the loss function w.r.t. the weights as MSE of the model predictions and the known outcomes
    loss = (weights, outcomes) -> mean_squared_error(model_train(weights), outcomes)

    # wrap the loss function via "grad" to be able to retrieve the gradients of the loss w.r.t. to the weights
    grad_loss = grad(loss)
    
    # get the gradients of the loss w.r.t. each weight parameter, i.e. this is a vector of length 4
    g = grad_loss(hidden_layer, train_output)
    
    # update the weights via gradient descend
    # "i_w" is the weight parameter index, "g_w" is the loss gradient of the weight
    for (i_w, g_w) in enumerate(g)
        weights[i_w] -= learningrate * g_w
    end

    # return the current model's loss
    return loss(weights, outcomes)
end

# randomly initialize the hidden layer weights
hidden_layer = randn(Float64, 4)

# apply the training; the "hidden_layer" weights are adjusted on each training iteration (epoch) to lower the loss
for i in 1:300
    current_loss = train(train_data, train_output, hidden_layer)
    println(i, ": loss = ", round(current_loss, digits=5), " weights = ", round.(hidden_layer, digits=4))
end

