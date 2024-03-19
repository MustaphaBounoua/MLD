import torch
import torch.nn as nn

# Nexus
class LinearEncoder(nn.Module):
    def __init__(self, name, input_dim, layer_sizes, output_dim):
        super(LinearEncoder, self).__init__()

        # Variables
        self.name = name
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.output_dim = output_dim

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
        self.fc_mu = nn.Linear(pre, output_dim)
        self.fc_logvar = nn.Linear(pre, output_dim)

        # Print information
        print('Info:' + str(self.name))
        print(f'Layers: {enc_layers}')
        self.network = nn.Sequential(*enc_layers)

    def forward(self, x):
        h = self.network(x)
        return self.fc_mu(h), self.fc_logvar(h)


class NexusEncoder(nn.Module):
    def __init__(self, name, input_dim, layer_sizes, output_dim):
        super(NexusEncoder, self).__init__()

        # Variables
        self.name = name
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.output_dim = output_dim

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
        self.fc_out = nn.Linear(pre, output_dim)

        # Print information
        print('Info:' + str(self.name))
        print(f'Layers: {enc_layers}')
        self.network = nn.Sequential(*enc_layers)

    def forward(self, x):
        h = self.network(x)
        return self.fc_out(h)
