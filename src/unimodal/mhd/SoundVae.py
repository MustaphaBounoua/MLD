import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

FRAME_SIZE = 512
CONTEXT_FRAMES = 32
SPECTROGRAM_BINS = FRAME_SIZE//2 + 1
eta =1e-6

path_sigma = "data/data_mhd/trained_models/sigma_vae.pth.tar"


class SoundEncoder(nn.Module):
    def __init__(self, latent_dim, deterministic =False):
        super(SoundEncoder, self).__init__()
        self.deterministic = deterministic
        # Properties
        self.conv_layer_0 = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Output layer of the network
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_logvar = nn.Linear(2048, latent_dim)

        # print(f'SoundEncoder: {[self.conv_layer, self.classifier]}')

    def forward(self, x):
        x = self.conv_layer_0(x)
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        h = x.view(x.size(0), -1)
        
        if self.deterministic:
            return self.fc_mu(h)
        else:
            return self.fc_mu(h), self.fc_logvar(h)

class MHDSoundEncoderPLus(nn.Module):

    def __init__(self, latent_dim,latent_dim_w, deterministic =False):
        super(MHDSoundEncoderPLus, self).__init__()
        self.deterministic = deterministic
        self.latent_dim = latent_dim
        # Properties
        self.conv_layer_0 = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )


        self.conv_layer_0_w = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_layer_1_w = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_layer_2_w = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Output layer of the network
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_logvar = nn.Linear(2048, latent_dim)

        self.fc_mu_w = nn.Linear(2048, latent_dim_w)
        self.fc_logvar_w = nn.Linear(2048, latent_dim_w)

        # print(f'SoundEncoder: {[self.conv_layer, self.classifier]}')

    def forward(self, data):

        x = self.conv_layer_0(data)
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        h = x.view(x.size(0), -1)
        mu_u= self.fc_mu(h)
        l_u= self.fc_logvar(h)


        x = self.conv_layer_0_w(data)
        x = self.conv_layer_1_w(x)
        x = self.conv_layer_2_w(x)
        h = x.view(x.size(0), -1)
        mu_w= self.fc_mu_w(h)
        l_w= self.fc_logvar_w(h)

        return mu_w, F.softmax(l_w, dim=-1) * l_w.size(-1) + eta, mu_u, F.softmax(l_u, dim=-1) * l_u.size(-1) + eta
   

        



class SoundDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(SoundDecoder, self).__init__()

        self.upsampler = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )

        self.hallucinate_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.hallucinate_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.hallucinate_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
        )


    def forward(self, z):
        z = self.upsampler(z)
        z = z.view(-1, 256, 8, 1)
        z = self.hallucinate_0(z)
        z = self.hallucinate_1(z)
        out = self.hallucinate_2(z)
        return torch.sigmoid(out)




class SoundDecoderPlus(nn.Module):
    def __init__(self, latent_dim):
        super(SoundDecoderPlus, self).__init__()
        
        self.upsampler = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )

        self.hallucinate_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.hallucinate_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.hallucinate_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
        )


    def forward(self, z_o):
        
        z =z_o.view(-1,z_o.size(-1))
        z = self.upsampler(z)
        
        z = z.view(-1, 256, 8, 1)
        z = self.hallucinate_0(z)
        z = self.hallucinate_1(z)
        out = self.hallucinate_2(z)

        if len(z_o.size()) == 2:
            out = out.view(*z_o.size()[:1], *out.size()[1:])
        else:
            out = out.view(*z_o.size()[:2], *out.size()[1:])
            
        return torch.sigmoid(out)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
    
    


def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor


class SigmaVAE(nn.Module):
    def __init__(self, latent_dim, use_cuda=False):

        super(SigmaVAE, self).__init__()

        # Parameters
        self.latent_dim = latent_dim
        self.use_cuda = use_cuda

        # Components
        self.mod_encoder = SoundEncoder(latent_dim=self.latent_dim)
        self.mod_decoder = SoundDecoder(latent_dim=self.latent_dim)


    def reparametrize(self, mu, logvar):

        # Sample epsilon from a random gaussian with 0 mean and 1 variance
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        # Check if cuda is selected
        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # std = exp(0.5 * log_var)
        std = logvar.mul(0.5).exp_()

        # z = std * epsilon + mu
        return mu.addcmul(std, epsilon)


    def forward(self, x):
        mu, logvar = self.mod_encoder(x)
        z = self.reparametrize(mu, logvar)
        out = self.mod_decoder(z)

        # Compute log_sigma optimal
        log_sigma = ((x - out) ** 2).mean([0, 1, 2, 3], keepdim=True).sqrt().log()

        # log_sigma = self.log_sigma
        out_log_sigma = softclip(log_sigma, -6)

        return out, out_log_sigma, z, mu, logvar
    
    
    
    
    
class SigmaSoundEncoder(nn.Module):
    def __init__(self, latent_dim):
        
        super(SigmaSoundEncoder, self).__init__()
        self.sigma_vae = SigmaVAE(latent_dim=128)
        self.sigma_vae.load_state_dict(torch.load(path_sigma)['state_dict'])
        
        
        self.linear_layer_mu = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
 
        )
       
        self.linear_layer_logvar = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
       
       
    def forward(self,x):
        with torch.no_grad():
            mu,logvar = self.sigma_vae.mod_encoder(x)
            
        return self.linear_layer_mu(mu), self.linear_layer_mu(logvar)
    
    
    
    
class SigmaSoundDecoder(nn.Module):
    def __init__(self, latent_dim):
        
        super(SigmaSoundDecoder, self).__init__()
        self.sigma_vae = SigmaVAE(latent_dim=128)
        self.sigma_vae.load_state_dict(torch.load(path_sigma)['state_dict'])
        
        
        self.linear_layer_z = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
       
       
       
    def forward(self,z):
        z = self.linear_layer_z(z)
        with torch.no_grad():
            reconstruct = self.sigma_vae.mod_decoder(z)
            
        
        return reconstruct    
     