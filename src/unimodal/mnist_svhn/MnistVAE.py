import torch
import torch.nn as nn
import torch.distributions as dist
from torch.nn import functional as F
import pytorch_lightning as pl


LATENT_DIM =64
INPUT_DATA_DIM = 784
dataSize = torch.Size([1, 28, 28])
num_hidden_layers = 1
eta = 1e-6


# FBASE = 32
# CHANNELS = 1

# class MnistEncoder(pl.LightningModule):

#     def __init__(self, latent_dim , input_size = INPUT_DATA_DIM ,channels= CHANNELS, deterministic = False):

#         super(MnistEncoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.channels = channels
#         self.deterministic = deterministic
#         self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=4, stride=2, padding=1, dilation=1);
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, dilation=1);
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, dilation=1);
#         self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0, dilation=1);
#         self.relu = nn.ReLU();
  
#         self.hidden_mu = nn.Linear(in_features=128, out_features=latent_dim, bias=True)
#         self.hidden_logvar = nn.Linear(in_features=128, out_features=latent_dim, bias=True)
#         # c1, c2 size: latent_dim x 1 x 1

#     def forward(self, x):
#         h = self.conv1(x);
#         h = self.relu(h);
#         h = self.conv2(h);
#         h = self.relu(h);
#         h = self.conv3(h)
#         h = self.relu(h)
#         h = self.conv4(h)
#         h = self.relu(h)
#         h = h.view(h.size(0), -1)
#         latent_space_mu = self.hidden_mu(h)
#         latent_space_logvar = self.hidden_logvar(h)
#         latent_space_mu = latent_space_mu.view(latent_space_mu.size(0), -1)
#         latent_space_logvar = latent_space_logvar.view(latent_space_logvar.size(0), -1)
#         if self.deterministic == True :
#             return latent_space_mu
#         else:
#             return latent_space_mu,latent_space_logvar




# class MnistDecoder(pl.LightningModule):

#     def __init__(self,input_size = INPUT_DATA_DIM, latent_dim = LATENT_DIM , channels =CHANNELS):

#         super(MnistDecoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.channels= channels
        
#         self.linear = nn.Linear(latent_dim, 128)
#         self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1, padding=0, dilation=1)
#         self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, dilation=1)
#         self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, dilation=1)
#         self.conv4 = nn.ConvTranspose2d(32, self.channels, kernel_size=4, stride=2, padding=1, dilation=1)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
        
    
#     def forward(self, x):
    
#         z = self.linear(x)
#         z = z.view(z.size(0), z.size(1), 1, 1)
#         x_hat = self.relu(z)
#         x_hat = self.conv1(x_hat)
#         x_hat = self.relu(x_hat)
#         x_hat = self.conv2(x_hat)
#         x_hat = self.relu(x_hat)
#         x_hat = self.conv3(x_hat)
#         x_hat = self.relu(x_hat)
#         x_hat = self.conv4(x_hat)
#         return self.sigmoid(x_hat)



















class MnistEncoder(nn.Module):
    def __init__(self, input_size = INPUT_DATA_DIM, latent_dim = LATENT_DIM, deterministic = False):
        super(MnistEncoder, self).__init__()
        self.deterministic = deterministic
        self.hidden_dim = 400;

        modules = []
        modules.append(nn.Sequential(nn.Linear(784, self.hidden_dim), nn.ReLU(True)))
        modules.extend([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))
                        for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.relu = nn.ReLU();
  
        self.hidden_mu = nn.Linear(in_features=self.hidden_dim, out_features= latent_dim, bias=True)
        self.hidden_logvar = nn.Linear(in_features=self.hidden_dim, out_features= latent_dim, bias=True)


    def forward(self, x):
        h = x.view(*x.size()[:-3], -1);
        h = self.enc(h);
        h = h.view(h.size(0), -1);

        latent_space_mu = self.hidden_mu(h);
        latent_space_logvar = self.hidden_logvar(h);
        latent_space_mu = latent_space_mu.view(latent_space_mu.size(0), -1);
        latent_space_logvar = latent_space_logvar.view(latent_space_logvar.size(0), -1);
        if self.deterministic == True:
            return latent_space_mu
        else:
            return latent_space_mu, latent_space_logvar;



class MnistDecoder(nn.Module):
    def __init__(self, input_size = INPUT_DATA_DIM, latent_dim = LATENT_DIM):
        super(MnistDecoder, self).__init__();
      
        self.hidden_dim = 400;
        modules = []
 
        modules.append(nn.Sequential(nn.Linear(latent_dim, self.hidden_dim), nn.ReLU(True)))

        modules.extend([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))
                        for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(self.hidden_dim, 784)
        self.relu = nn.ReLU();
        self.sigmoid = nn.Sigmoid();

    def forward(self, z):
        x_hat = self.dec(z);
        x_hat = self.fc3(x_hat);
        x_hat = self.sigmoid(x_hat)
        x_hat = x_hat.view(*z.size()[:-1], *dataSize)
        return x_hat
















# class MnistEncoderplus(nn.Module):
#     def __init__(self,latent_dim,latent_dim_w ):
#         super(MnistEncoderplus, self).__init__()
#         self.enc_w = MnistEncoder(latent_dim=latent_dim_w)
#         self.enc_u = MnistEncoder(latent_dim=latent_dim)
    
#     def forward(self, x):
#         mu_w,l_w = self.enc_w(x)
#         mu_u,l_u= self.enc_u(x)
#         return mu_w, F.softmax(l_w, dim=-1) * l_w.size(-1) + eta, mu_u, F.softmax(l_u, dim=-1) * l_u.size(-1) + eta
   
   


class MnistEncoderplus(nn.Module):
    def __init__(self, latent_dim,latent_dim_w, input_size = INPUT_DATA_DIM):
        super(MnistEncoderplus, self).__init__()

        self.hidden_dim = 400;
        self.latent_dim = latent_dim
        modules = []
        modules.append(nn.Sequential(nn.Linear(784, self.hidden_dim), nn.ReLU(True)))
        modules.extend([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))
                        for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.relu = nn.ReLU();
  
        self.hidden_mu = nn.Linear(in_features=self.hidden_dim, out_features= latent_dim + latent_dim_w, bias=True)
        self.hidden_logvar = nn.Linear(in_features=self.hidden_dim, out_features= latent_dim +latent_dim_w, bias=True)


    def forward(self, x):
        h = x.view(*x.size()[:-3], -1);
        h = self.enc(h);
        h = h.view(h.size(0), -1);

        latent_space_mu = self.hidden_mu(h);
        latent_space_logvar = self.hidden_logvar(h);
        latent_space_mu = latent_space_mu.view(latent_space_mu.size(0), -1);
        latent_space_logvar = latent_space_logvar.view(latent_space_logvar.size(0), -1);
        
        mu_u  = latent_space_mu[:,:self.latent_dim]
        l_u =latent_space_logvar[:,:self.latent_dim]

        mu_w = latent_space_mu[:,self.latent_dim:]
        l_w =latent_space_logvar[:,self.latent_dim:]

        return mu_w, F.softmax(l_w, dim=-1) * l_w.size(-1) + eta, mu_u, F.softmax(l_u, dim=-1) * l_u.size(-1) + eta
   