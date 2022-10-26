# https=>//machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/


module NeuralNetsLux end

using Random
Random.seed!(12344)

function initialize_network(n_inputs, n_hidden, n_outputs)
    network = []
    hidden_layer = [Dict{String,Any}("weights" => [rand() for i = 1:n_inputs+1]) for i = 1:n_hidden]
    push!(network, hidden_layer)
    output_layer = [Dict{String,Any}("weights" => [rand() for i = 1:n_hidden+1]) for i = 1:n_outputs]
    push!(network, output_layer)
    return network
end

network = initialize_network(2, 1, 2)
for layer in network
    print(layer)
end

##############################################################################################

# Calculate neuron activation for an input
function activate(weights, inputs)
    activation = weights[end]
    for i in axes(weights, 1)[1:end-1]
        activation += weights[i] * inputs[i]
    end
    return activation
end

# Transfer neuron activation
function transfer(activation)
    return @. 1.0 / (1.0 + exp(-activation))
end

# Forward propagate input to a network output
function forward_propagate(network, row)
    inputs = row
    for layer in network
        new_inputs = []
        for neuron in layer
            activation = activate(neuron["weights"], inputs)
            neuron["output"] = transfer(activation)
            push!(new_inputs, neuron["output"])
        end
        inputs = new_inputs
    end
    return inputs
end

# test forward propagation
network = [[Dict{String,Any}("weights"=> [0.13436424411240122, 0.8474337369372327, 0.763774618976614])],
		  [Dict{String,Any}("weights"=> [0.2550690257394217, 0.49543508709194095]), Dict{String,Any}("weights"=> [0.4494910647887381, 0.651592972722763])]]
row = [1, 0, nothing]
output = forward_propagate(network, row)
print(output)


# Calculate the derivative of an neuron output
function transfer_derivative(output)
	return output * (1.0 - output)
end

# Backpropagate error and store in neurons
function backward_propagate_error(network, expected)
	for i in reversed(range(len(network)))
		layer = network[i]
		errors = list()
		if i != length(network)-1
			for j in range(len(layer))
				error = 0.0
				for neuron in network[i+1]
					error += (neuron["weights"][j] * neuron["delta"])
                end
				push!(errors,error)
            end
		else
			for j in 1:length(layer)
				neuron = layer[j]
				push!(errors,neuron["output"] - expected[j])
            end
        end
		for j in 1:length(layer)
			neuron = layer[j]
			neuron["delta"] = errors[j] * transfer_derivative(neuron["output"])
        end
    end
end
