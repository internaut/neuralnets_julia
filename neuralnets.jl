# Neural Network from Scratch in Julia
# adapted from https://sirupsen.com/napkin/neural-net
#
# May 2022
#
# author: Markus Konrad <post@mkonrad.net>
#

using LinearAlgebra  # for dot

# first goal: determine the "brightness" of a 2x2 pixels image using a simple "neural network"
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
train_data = Vector{Vector{Float64}}(undef, N_TRAINING_SAMPLES)   # this is probably not the right prealloc.
train_output = Vector{Float64}(undef, N_TRAINING_SAMPLES)

for i in 1:N_TRAINING_SAMPLES
    d = rand(Float64, 4)   # random vector with values in [0, 1)
    train_data[i] = d
    train_output[i] = mean(d)
end

# we define a loss function that measures the quality of the current model; we use the mean squared error

# x are the actual values, x_hat the predicted values
# note the dots before "-" and "^" operations; they denote that the elementwise operations (the inputs are vectors)
# should be performed
mean_squared_error = (x_hat, x) -> mean((x .- x_hat).^2)


model = (input_data, weights) -> dot(input_data, weights)

function train(data, weights)
    outputs = Vector{Float64}(undef, length(data))
    for (i, d) in enumerate(data)
        outputs[i] = model(d, weights)
    end

    outputs
end

predictions = train(train_data, [0.98, 0.4, 0.86, -0.08])

# quite bad MSE for the above weights:
mean_squared_error(predictions, train_output)

