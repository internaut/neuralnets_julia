# Example of using AutoGrad for linear regression of a simple model y ~ wx + b
#
# May 2022
#
# author: Markus Konrad <post@mkonrad.net>
#

using AutoGrad

# defines the linear model with parameters slope "w" and intercept "b", i.e. y ~ wx + b
lm = (w, b, x) -> w .* x .+ b

# try it out with example model x -> 2x+1:
lm(2, 1, collect(-3:3))

# generate some training data for an example model y ~ 2x + 1 
n_samples = 10
true_slope = 2.0
true_intercept = 1.0
xs = Vector{Float64}(0:(n_samples-1))
ys = true_intercept .+ true_slope .* xs .+ randn(n_samples) / 10   # we add a random error

# the goal is to recover the true_slope and true_intercept from the generated data
# define the linear model as function of slope "w" and intercept "b"; this is very important because we want to get
# the gradients of the loss with respect to "w" and "b"; "x" is fixed as input data
lm_x = w_b -> lm(w_b[1], w_b[2], xs)

# define the loss of the linear model as sum of squared errors again w.r.t. parameters "w" and "b"
loss = (w_b, y) -> sum((lm_x(w_b) .- y).^2)

# now AutoGrad comes into play: wrap the loss function via "grad" to be able to retrieve the gradients of the loss
# w.r.t. parameters "w" and "b"
grad_loss = grad(loss)

# start with some initial values for the parameters
w = 0.0
b = 0.0

# now we apply stochastic gradient descend (SGD) with a learning rate "lr"; it's important to not set this rate
# too high, because otherwise we will miss the local minimum of the loss; keep in mind that the input data is not
# normalized in any way; we should normally do that before applying SGD

lr = 0.001
for i in 1:1000
    println("loop ", i)
    
    # calculate the loss (this is just for information)
    println("> loss = ", loss([w, b], ys))

    # get the gradients of the loss w.r.t. each parameter
    g = grad_loss([w, b], ys)
    
    # descend by learning rate and respective gradient
    w -= lr * g[1]
    b -= lr * g[2]

    println("> w = ", w)
    println("> b = ", b)
end

# show final model
println("final model: y ~ ", round(w, digits=3), "x + ", round(b, digits=3))
final_loss = loss([w, b], ys)
println("final loss (sum of squared errors): ", round(final_loss, digits=3))
println("RMSE: ", round(sqrt(final_loss / n_samples), digits=3))
