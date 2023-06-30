import torch
import torch.nn as nn
import optuna
import all_inputs

input_size = all_inputs.input_size
output_size = all_inputs.output_size

def define_model(params):
# We optimize the number of layers, hidden units and dropout ratio in each layer.
  n_layers = int(params['n_layers'])
  layers = []
  hidden_in = input_size
  for i in range(n_layers):
    hidden_out = int(params["n_hidden{}".format(i)])
    layers.append(nn.Linear(hidden_in, hidden_out))
    layers.append(nn.LeakyReLU(0.2))
    p = params["dropoutprob_{}".format(i)]
    layers.append(nn.Dropout(p))
    hidden_in = hidden_out

  layers.append(nn.Linear(hidden_in, output_size))

  return nn.Sequential(*layers)