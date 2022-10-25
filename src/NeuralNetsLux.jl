# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/


module NeuralNetsLux
end

using Random
Random.seed!(12344)

function initialize_network(n_inputs, n_hidden, n_outputs)
    network = []
    hidden_layer = [Dict("weights"=>[rand() for i in 1:n_inputs+1]) for i in 1:n_hidden]
    push!(network,hidden_layer)
    output_layer = [Dict("weights"=>[rand() for i in 1:n_hidden+1]) for i in 1:n_outputs]
    push!(network,output_layer)
    return network
end

network = initialize_network(2, 1, 2)
for layer in network
    print(layer)
end

# Calculate neuron activation for an input
function activate(weights, inputs)
    activation = weights[end]
    for i in axes(weights,1)
        activation += weights[i] * inputs[i]
    end
    return activation
end

# Transfer neuron activation
function transfer(activation)
    return 1.0 / (1.0 + exp(-activation))
end
