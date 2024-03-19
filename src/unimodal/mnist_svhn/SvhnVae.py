import torch
import torch.nn as nn
import torch.distributions as dist
from torch.nn import functional as F
import pytorch_lightning as pl


LATENT_DIM = 64
INPUT_DATA_DIM = 3* 32* 32
FBASE = 32
CHANNELS = 3
eta = 1e-6





class SvhnEncoder_plus(pl.LightningModule):

    def __init__(self, latent_dim ,latent_dim_w=None , input_size = INPUT_DATA_DIM ,channels= CHANNELS, deterministic = False):

        super(SvhnEncoder_plus, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.deterministic = deterministic
        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=4, stride=2, padding=1, dilation=1);
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, dilation=1);
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, dilation=1);
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0, dilation=1);
        self.relu = nn.ReLU();
  
        self.hidden_mu = nn.Linear(in_features=128, out_features=latent_dim +latent_dim_w, bias=True)
        self.hidden_logvar = nn.Linear(in_features=128, out_features=latent_dim+latent_dim_w, bias=True)
        # c1, c2 size: latent_dim x 1 x 1

    def forward(self, x):
        h = self.conv1(x);
        h = self.relu(h);
        h = self.conv2(h);
        h = self.relu(h);
        h = self.conv3(h)
        h = self.relu(h)
        h = self.conv4(h)
        h = self.relu(h)
        h = h.view(h.size(0), -1)
       
     
        latent_space_mu = self.hidden_mu(h)
        latent_space_logvar = self.hidden_logvar(h)
        latent_space_mu = latent_space_mu.view(latent_space_mu.size(0), -1)
        latent_space_logvar = latent_space_logvar.view(latent_space_logvar.size(0), -1)

        mu_u  = latent_space_mu[:,:self.latent_dim]
        l_u =latent_space_logvar[:,:self.latent_dim]

        mu_w = latent_space_mu[:,self.latent_dim:]
        l_w =latent_space_logvar[:,self.latent_dim:]

        return mu_w, F.softmax(l_w, dim=-1) * l_w.size(-1) + eta, mu_u, F.softmax(l_u, dim=-1) * l_u.size(-1) + eta




















fBase = 32

class SvhnEncoder(pl.LightningModule):

    def __init__(self, latent_dim , input_size = INPUT_DATA_DIM ,channels= CHANNELS, deterministic = False):

        super(SvhnEncoder, self).__init__()
        self.deterministic = deterministic
        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(3, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
        )
        self.c1 = nn.Conv2d(fBase * 4, latent_dim, 4, 1, 0, bias=True)
        if self.deterministic == False:
            self.c2 = nn.Conv2d(fBase * 4, latent_dim, 4, 1, 0, bias=True)

    def forward(self, x):
        e = self.enc(x)
        if self.deterministic:
        # lv = self.c2(e).squeeze()
            return self.c1(e).squeeze()
        else:
            return self.c1(e).squeeze(),self.c2(e).squeeze()




class SvhnDecoder(pl.LightningModule):

    def __init__(self,input_size = INPUT_DATA_DIM, latent_dim = LATENT_DIM , channels =CHANNELS):

        super(SvhnDecoder, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, fBase * 4, 4, 1, 0, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.ConvTranspose2d(fBase, 3, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # Output size: 3 x 32 x 32
        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        out = out.view(*z.size()[:-3], *out.size()[1:])
        return out



