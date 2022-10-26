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
function forward_propagate!(network, row)
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


# Calculate the derivative of an neuron output
function transfer_derivative(output)
	return output * (1.0 - output)
end

# Backpropagate error and store in neurons
function backward_propagate_error!(network, expected)
	for i in reverse(eachindex(network))
		layer = network[i]
		errors = []
		if i != length(network)
			for j in eachindex(layer)
				error = 0.0
				for neuron in network[i+1]
					error += (neuron["weights"][j] * neuron["delta"])
                end
				push!(errors,error)
            end
		else
			for j in eachindex(layer)
				neuron = layer[j]
				push!(errors,neuron["output"] - expected[j])
            end
        end
		for j in eachindex(layer)
			neuron = layer[j]
			neuron["delta"] = errors[j] * transfer_derivative(neuron["output"])
        end
    end
end

# test backpropagation of error
network = [[Dict{String,Any}("output"=> 0.7105668883115941, "weights"=> [0.13436424411240122, 0.8474337369372327, 0.763774618976614])],
		[Dict{String,Any}("output"=> 0.6213859615555266, "weights"=> [0.2550690257394217, 0.49543508709194095]), Dict{String,Any}("output"=> 0.6573693455986976, "weights"=> [0.4494910647887381, 0.651592972722763])]]
expected = [0.0, 1.0]
backward_propagate_error!(network, expected)
for layer in network
	print(layer)
end


# Update network weights with error
function update_weights!(network, row, l_rate)
	for i in eachindex(network)
		inputs = row[begin:end-1]
		if i != 0
			inputs = [neuron["output"] for neuron in network[i - 1]]
        end
		for neuron in network[i]
			for j in eachindex(inputs)
				neuron["weights"][j] -= l_rate * neuron["delta"] * inputs[j]
            end
			neuron["weights"][-1] -= l_rate * neuron["delta"]
        end
    end
    network
end
using Printf

# Train a network for a fixed number of epochs
function train_network!(network, train, l_rate, n_epoch, n_outputs)
	for epoch in 1:n_epoch
		sum_error = 0
		for row in train
			outputs = forward_propagate!(network, row)
			expected = [0 for i in 1:n_outputs]
			expected[Int(row[end]+1)] = 1
			sum_error += sum([(expected[i]-outputs[i])^2 for i in eachindex(expected))])
			backward_propagate_error!(network, expected)
			update_weights!(network, row, l_rate)
		@printf(">epoch=%d, lrate=%.3f, error=%.3f", epoch, l_rate, sum_error)
        end
    end
end

# Test training backprop algorithm

dataset = [[2.7810836,2.550537003,0],
        [1.465489372,2.362125076,0],
        [3.396561688,4.400293529,0],
        [1.38807019,1.850220317,0],
        [3.06407232,3.005305973,0],
        [7.627531214,2.759262235,1],
        [5.332441248,2.088626775,1],
        [6.922596716,1.77106367,1],
        [8.675418651,-0.242068655,1],
        [7.673756466,3.508563011,1]]
n_inputs = length(dataset[1]) - 1
n_outputs = length(Set([row[end] for row in dataset]))
network = initialize_network(n_inputs, 2, n_outputs)
train_network!(network, dataset, 0.5, 20, n_outputs)
for layer in network
	print(layer)
end
