import os
import torch
import torch.nn as nn
import numpy as np
import cma
import json
import time

EVAL_GAME_COUNT = 1
LOSS = 0
WIN = 1
DRAW = 2
CONTINUE = 3

# Define the neural network
class ConvNetwork(nn.Module):
    def __init__(self, input_channels, output_size):
        super(ConvNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1) # Convolutional Layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Fully connected layer (adjust size)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten before fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Objective function for CMA-ES optimization
def objective_function(input_file, output_file, parameters, model):
    """
    Compute loss for CMA-ES.
    """
    # Load parameters into the model
    params_tensor = torch.tensor(parameters, dtype=torch.float32)
    offset = 0
    for param in model.parameters():
        num_params = param.numel()
        param.data = params_tensor[offset:offset + num_params].view(param.size())
        offset += num_params

    eval_game_count = EVAL_GAME_COUNT
    game_count = 0
    loss = 10000000
    while game_count < eval_game_count:
        inputs, game_end = load_inputs(input_file)
        if game_end == LOSS or game_end == WIN or game_end == DRAW:
            game_count += 1
            # set loss 
        else:
            # Forward pass
            outputs = model(inputs)
            #loss = ((outputs - target_outputs) ** 2).mean().item()
            write_output(output_file, outputs)
        

    return loss

# Load input vector from a file
def load_inputs(input_file):
    loading = True
    while loading:
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
            inputs = torch.tensor(data["inputs"], dtype=torch.float32)
            game_end = data["GameEnd"]
            #targets = torch.tensor(data["targets"], dtype=torch.float32)
            loading = False
            os.remove(input_file)
        except:
            print("Still waiting")
            time.sleep(1)
    return inputs, game_end

# Save output vector to a file
def write_output(output_file, outputs):
    outputs_dict = {"outputs": outputs.tolist()}
    with open(output_file, 'w') as f:
        json.dump(outputs_dict, f)

# Main function
def main(input_file, output_file, input_size, output_size, population_size, max_generations):

    # Initialize the model
    model = ConvNetwork(input_size, output_size)

    # Flatten model parameters
    initial_params = torch.cat([p.flatten() for p in model.parameters()]).detach().numpy()

    # Initialize CMA-ES
    optimizer = cma.CMAEvolutionStrategy(initial_params, 0.5, {'popsize': population_size})

    # Optimization loop
    for generation in range(max_generations):
        solutions = optimizer.ask()
        losses = [objective_function(input_file, output_file, sol, model) for sol in solutions]
        optimizer.tell(solutions, losses)

        # Print progress
        print(f"Generation {generation + 1}/{max_generations}, Best Loss: {min(losses):.6f}")

    # Get best parameters
    best_parameters = optimizer.result.xbest

    # Load best parameters into the model
    offset = 0
    for param in model.parameters():
        num_params = param.numel()
        param.data = torch.tensor(best_parameters[offset:offset + num_params], dtype=torch.float32).view(param.size())
        offset += num_params

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize a PyTorch network using CMA-ES.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input file (JSON).")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output file (JSON).")
    parser.add_argument('--input_size', type=int, required=True, help="Input size of the network.")
    parser.add_argument('--output_size', type=int, required=True, help="Output size of the network.")
    parser.add_argument('--population_size', type=int, default=50, help="Population size for CMA-ES.")
    parser.add_argument('--max_generations', type=int, default=100, help="Maximum number of generations for CMA-ES.")
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.input_size, args.output_size, args.population_size, args.max_generations)
