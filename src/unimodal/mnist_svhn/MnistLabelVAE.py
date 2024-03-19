import torch
import torch.nn as nn


INPUT_DATA_DIM = 8*71
LATENT_DIM = 10
NUM_FEATURE = 71
DIM = 64


class FeatureEncText(nn.Module):
    def __init__(self, dim, num_features):
        super(FeatureEncText, self).__init__()
        
        self.dim = dim
        self.conv1 = nn.Conv1d(num_features, 2*self.dim, kernel_size=1);
        self.conv2 = nn.Conv1d(2*self.dim, 2*self.dim, kernel_size=4, stride=2, padding=1, dilation=1);
        self.conv5 = nn.Conv1d(2*self.dim, 2*self.dim, kernel_size=4, stride=2, padding=0, dilation=1);
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(-2,-1);
        out = self.conv1(x);
        out = self.relu(out);
        out = self.conv2(out);
        out = self.relu(out);
        out = self.conv5(out);
        out = self.relu(out);
        h = out.view(-1, 2*self.dim)
        return h;


class TextEncoder(nn.Module):
    def __init__(self, num_features=NUM_FEATURE, dim = DIM, latent_dim=LATENT_DIM,deterministic=False):
        super(TextEncoder, self).__init__()
        self.deterministic =deterministic
        self.text_feature_enc = FeatureEncText(dim, num_features);
        #non-factorized
        self.latent_mu = nn.Linear(in_features=2*dim, out_features=latent_dim, bias=True)
        self.latent_logvar = nn.Linear(in_features=2*dim, out_features=latent_dim, bias=True)


    def forward(self, x):
        h = self.text_feature_enc(x)
        latent_space_mu = self.latent_mu(h)
        latent_space_logvar = self.latent_logvar(h)
        if self.deterministic == True:
            return latent_space_mu
        else:
            return latent_space_mu, latent_space_logvar


class TextDecoder(nn.Module):
    def __init__(self, num_features=NUM_FEATURE, dim = DIM, latent_dim=LATENT_DIM):
        super(TextDecoder, self).__init__()

        self.linear = nn.Linear(latent_dim, 2*dim)
        self.conv1 = nn.ConvTranspose1d(2*dim, 2*dim,
                                        kernel_size=4, stride=1, padding=0, dilation=1);
        self.conv2 = nn.ConvTranspose1d(2*dim, 2*dim,
                                        kernel_size=4, stride=2, padding=1, dilation=1);
        self.conv_last = nn.Conv1d(2*dim, num_features, kernel_size=1);
        self.relu = nn.ReLU()
        self.out_act = nn.LogSoftmax(dim=-2);

    def forward(self, z):
        z = self.linear(z)
        x_hat = z.view(z.size(0), z.size(1), 1)
        x_hat = self.conv1(x_hat)
        x_hat = self.relu(x_hat)
        x_hat = self.conv2(x_hat)
        x_hat = self.relu(x_hat)
        x_hat = self.conv_last(x_hat)
        log_prob = self.out_act(x_hat)
        log_prob = log_prob.transpose(-2, -1)
        return log_prob
