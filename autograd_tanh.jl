# Example of using AutoGrad on a user defined function.
# Taken and adapted from https://github.com/HIPS/autograd.
#
# May 2022
#
# author: Markus Konrad <post@mkonrad.net>
#


using AutoGrad

# user defined function
function tanh(x)
    y = exp(-2.0 * x)
    (1.0 - y) / (1.0 + y)
end

# wrapper to calculate gradients
grad_tanh = grad(tanh)

# parameter value at which to evaluate the function
x = 1.0

# get the gradient function (derivative)
grad_tanh(x)

# check via approximation using a finite difference
ϵ = 0.0001
(tanh(x + ϵ) - tanh(x - ϵ)) / 2ϵ