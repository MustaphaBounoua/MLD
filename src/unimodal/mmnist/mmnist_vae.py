import torch
import torch.nn as nn
from torch.nn import functional as F
eta = 1e-6

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(torch.nn.Module):
    def __init__(self, ndims):
        super(Unflatten, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        return x.view(x.size(0), *self.ndims)
    
class EncoderImg(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, latent_dim, deterministic = False):
        super(EncoderImg, self).__init__()

        self.latent_dim = latent_dim
        self.deterministic = deterministic
        
        self.shared_encoder = nn.Sequential(                          # input shape (3, 28, 28)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),     # -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # -> (128, 4, 4)
            nn.ReLU(),
            Flatten(),                                                # -> (2048)
            nn.Linear(2048, latent_dim),       # -> (ndim_private + ndim_shared)
            nn.ReLU(),
        )

        # content branch
        self.class_mu = nn.Linear(self.latent_dim , self.latent_dim )
        self.class_logvar = nn.Linear(self.latent_dim , self.latent_dim )
        # # optional style branch
        # if flags.factorized_representation:
        #     self.style_mu = nn.Linear(flags.style_dim + flags.class_dim, flags.style_dim)
        #     self.style_logvar = nn.Linear(flags.style_dim + flags.class_dim, flags.style_dim)

    def forward(self, x):
        h = self.shared_encoder(x)
        
        if self.deterministic :
            return self.class_mu(h)
        else : 
            return  self.class_mu(h), self.class_logvar(h)
        
        # if self.flags.factorized_representation:
        #     return self.style_mu(h), self.style_logvar(h), self.class_mu(h), \
        #            self.class_logvar(h)
        # else:
        #     return None, None, self.class_mu(h), self.class_logvar(h)

class MMNISTEncoderplus(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, latent_dim,latent_dim_w):
        super(MMNISTEncoderplus, self).__init__()

        self.latent_dim = latent_dim
    
        self.shared_encoder = nn.Sequential(                          # input shape (3, 28, 28)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),     # -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # -> (128, 4, 4)
            nn.ReLU(),
            Flatten(),                                                # -> (2048)
            nn.Linear(2048, latent_dim +latent_dim_w ),       # -> (ndim_private + ndim_shared)
            nn.ReLU(),
        )
        
        # content branch
        self.class_mu = nn.Linear(self.latent_dim +latent_dim_w , self.latent_dim +latent_dim_w )
        self.class_logvar = nn.Linear(self.latent_dim +latent_dim_w , self.latent_dim +latent_dim_w )

        # # optional style branch
        # if flags.factorized_representation:
        #     self.style_mu = nn.Linear(flags.style_dim + flags.class_dim, flags.style_dim)
        #     self.style_logvar = nn.Linear(flags.style_dim + flags.class_dim, flags.style_dim)


    def forward(self, x):
        h = self.shared_encoder(x)
        
        latent_space_mu =self.class_mu(h)
        latent_space_logvar = self.class_logvar(h)
        
        mu_u  = latent_space_mu[:,:self.latent_dim]
        l_u =latent_space_logvar[:,:self.latent_dim]

        mu_w = latent_space_mu[:,self.latent_dim:]
        l_w =latent_space_logvar[:,self.latent_dim:]

        return mu_w, F.softmax(l_w, dim=-1) * l_w.size(-1) + eta, mu_u, F.softmax(l_u, dim=-1) * l_u.size(-1) + eta
   

# class MMNISTEncoderplus(nn.Module):
#     def __init__(self,latent_dim,latent_dim_w ):
#         super(MMNISTEncoderplus, self).__init__()
#         self.enc_w = EncoderImg(latent_dim=latent_dim_w)
#         self.enc_u = EncoderImg(latent_dim=latent_dim)
    
#     def forward(self, x):
#         mu_w,l_w = self.enc_w(x)
#         mu_u,l_u= self.enc_u(x)

    
        
   
   


    


class DecoderImg(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, latent_dim):
        super(DecoderImg, self).__init__()
        self.latent_dim = latent_dim
        self.decoder = nn.Sequential(
            nn.Linear( self.latent_dim, 2048),                                # -> (2048)
            nn.ReLU(),
            Unflatten((128, 4, 4)),                                                            # -> (128, 4, 4)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),                   # -> (64, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (3, 28, 28)
        )

    def forward(self, class_latent_space):
        # if self.flags.factorized_representation:
        #     z = torch.cat((style_latent_space, class_latent_space), dim=1)
        # else:

        z = class_latent_space
        x_hat = self.decoder(z)
        # x_hat = torch.sigmoid(x_hat)
        return x_hat #, torch.tensor(0.75).to(z.device)  # NOTE: consider learning scale param, too



class DecoderImgplus(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, latent_dim):
        super(DecoderImgplus, self).__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear( self.latent_dim, 2048)
        self.act = nn.ReLU()
        Unflatten((128, 4, 4))  
        self.decoder = nn.Sequential(                                                       # -> (128, 4, 4)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),                   # -> (64, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (3, 28, 28)
        )

    def forward(self, z):
        # if self.flags.factorized_representation:
        #     z = torch.cat((style_latent_space, class_latent_space), dim=1)
        # else:
        class_latent_space =z
        out = self.fc(class_latent_space).view(-1, 128, 4, 4)

        out = self.decoder(out)
        if len(z.size()) == 2:
            out = out.view(*z.size()[:1], *out.size()[1:])
        else:
            out = out.view(*z.size()[:2], *out.size()[1:])
        # x_hat = torch.sigmoid(x_hat)
        return out #, torch.tensor(0.75).to(z.device)  # NOTE: consider learning scale param, too
