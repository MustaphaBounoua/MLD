import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Bernoulli
import numpy as np
from functools import reduce
import torch.nn.functional as F


INPUT_DIM = 10
LAYER_SIZE = [128, 128, 128]
symbol_mod_latent_dim = 5

# Symbol
class LabelEncoder(nn.Module):
    def __init__(self,latent_dim, name="", input_dim= INPUT_DIM, layer_sizes = LAYER_SIZE ,deterministic =False):
        super(LabelEncoder, self).__init__()

        # Variables
        self.name = name
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.output_dim = latent_dim
        self.deterministic = deterministic
        # Create Network
        enc_layers = []
        pre = input_dim

        for i in range(len(layer_sizes)):
            pos = layer_sizes[i]
            enc_layers.append(nn.Linear(pre, pos))
            enc_layers.append(nn.BatchNorm1d(pos))
            enc_layers.append(nn.LeakyReLU())

            # Check for input transformation
            pre = pos

        # Output layer of the network
        self.fc_mu = nn.Linear(pre, latent_dim)
        self.fc_logvar = nn.Linear(pre, latent_dim)

        # Print information
        print('Info:' + str(self.name))
        print(f'Layers: {enc_layers}')
        self.network = nn.Sequential(*enc_layers)

    def forward(self, x):
        h = self.network(x)
        if self.deterministic:
            return self.fc_mu(h)
        else:
            return self.fc_mu(h), self.fc_logvar(h)


class LabelDecoder(nn.Module):
    def __init__(self,latent_dim, name="", layer_sizes = np.flip(LAYER_SIZE), output_dim= INPUT_DIM):
        super(LabelDecoder, self).__init__()

        # Variables
        self.name = name
        self.id = id
        self.input_dim = latent_dim
        self.layer_sizes = layer_sizes
        self.output_dim = output_dim

        # Create Network
        dec_layers = []
        pre = latent_dim

        for i in range(len(layer_sizes)):
            pos = layer_sizes[i]

            # Check for input transformation
            dec_layers.append(nn.Linear(pre, pos))
            dec_layers.append(nn.BatchNorm1d(pos))
            dec_layers.append(nn.LeakyReLU())

            # Check for input transformation
            pre = pos

        dec_layers.append(nn.Linear(pre, output_dim))
        self.network = nn.Sequential(*dec_layers)

        # Print information
        print('Info:' + str(self.name))
        print(f'Layers: {dec_layers}')


    def forward(self, x):
        return self.network(x)  # No Log Softmax on output!