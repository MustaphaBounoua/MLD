import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM = 200
LAYER_SIZE = [512, 512, 512]
eta =1e-6
# Symbol
class TrajectoryEncoder(nn.Module):
    def __init__(self, latent_dim,input_dim = INPUT_DIM, layer_sizes =LAYER_SIZE, deterministic =False ):
        super(TrajectoryEncoder, self).__init__()

        # Variables
        self.deterministic = deterministic
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.output_dim = latent_dim

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
        self.fc_mu = nn.Linear(pre, self.output_dim)
        self.fc_logvar = nn.Linear(pre, self.output_dim)

        # Print information
    
 
        self.network = nn.Sequential(*enc_layers)

    def forward(self, x):
       # print(x)
        h = self.network(x)
  
        if self.deterministic:
            return self.fc_mu(h)
        else:
            return self.fc_mu(h), self.fc_logvar(h)



class MHDTrajEncoderPLus(nn.Module):
    def __init__(self, latent_dim,latent_dim_w,input_dim = INPUT_DIM, layer_sizes =LAYER_SIZE, deterministic =False ):
        super(MHDTrajEncoderPLus, self).__init__()

        # Variables
        self.deterministic = deterministic
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.output_dim = latent_dim
        self.latent_dim = latent_dim

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


        enc_layers_w = []
        pre_w = input_dim

        for i in range(len(layer_sizes)):
            pos = layer_sizes[i]
            enc_layers_w.append(nn.Linear(pre_w, pos))
            enc_layers_w.append(nn.BatchNorm1d(pos))
            enc_layers_w.append(nn.LeakyReLU())

            # Check for input transformation
            pre_w = pos


        # Output layer of the network
        self.fc_mu = nn.Linear(pre, latent_dim)
        self.fc_logvar = nn.Linear(pre, latent_dim)


        self.fc_mu_w = nn.Linear(pre_w, latent_dim_w)
        self.fc_logvar_w = nn.Linear(pre_w, latent_dim_w)

        # Printinformation
    
        self.network = nn.Sequential(*enc_layers)
        self.network_w = nn.Sequential(*enc_layers_w)

    def forward(self, x):
       # print(x)
        h = self.network(x)
        h_w = self.network_w(x)

        mu_u =self.fc_mu(h)
        l_u=self.fc_logvar(h)
        
        mu_w = self.fc_mu_w(h_w)
        l_w =self.fc_logvar_w(h_w)

        return mu_w, F.softmax(l_w, dim=-1) * l_w.size(-1) + eta, mu_u, F.softmax(l_u, dim=-1) * l_u.size(-1) + eta


      

# class MHDTrajEncoderPLus(nn.Module):
#     def __init__(self,latent_dim,latent_dim_w ):
#         super(MHDTrajEncoderPLus, self).__init__()
#         self.enc_w = TrajectoryEncoder(latent_dim=latent_dim_w)
#         self.enc_u = TrajectoryEncoder(latent_dim=latent_dim)
    
#     def forward(self, x):
#         mu_w,l_w = self.enc_w(x)
#         mu_u,l_u= self.enc_u(x)

    
#         return mu_w, F.softmax(l_w, dim=-1) * l_w.size(-1) + eta , mu_u , F.softmax(l_u, dim=-1) * l_u.size(-1) + eta
   

class TrajectoryDecoder(nn.Module):
    def __init__(self, latent_dim, input_dim = INPUT_DIM, layer_sizes =LAYER_SIZE):
        super(TrajectoryDecoder, self).__init__()

        # Variables
   
  
        self.input_dim = latent_dim
        self.layer_sizes = layer_sizes
        self.output_dim = input_dim

        # Create Network
        dec_layers = []
        pre =  self.input_dim


        for i in range(len(layer_sizes)):
            pos = layer_sizes[i]

            # Check for input transformation
            dec_layers.append(nn.Linear(pre, pos))
            dec_layers.append(nn.BatchNorm1d(pos))
            dec_layers.append(nn.LeakyReLU())

            # Check for input transformation
            pre = pos

        dec_layers.append(nn.Linear(pre,  self.output_dim))
        self.network = nn.Sequential(*dec_layers)






    def forward(self, z):
        out = z.view(-1,z.size(-1))
        out = self.network(out)
        if len(z.size()) == 2:
            out = out.view(*z.size()[:1], *out.size()[1:])
        else:
            out = out.view(*z.size()[:2], *out.size()[1:])
            
        return torch.sigmoid(out)










