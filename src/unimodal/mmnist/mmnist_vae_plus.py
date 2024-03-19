# PolyMNIST model specification

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import numpy as np



# Constants
dataSize = torch.Size([3, 28, 28])
# imgChans = dataSize[0]
# fBase = 32  # base size of filter channels
eta = 1e-6


def actvn(x):
    out = torch.nn.functional.leaky_relu(x, 2e-1)
    return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


# Classes
class Enc(nn.Module):
    """ Generate latent parameters for SVHN image data. """
    #   latent_dim, deterministic = False
    def __init__(self, latent_dim,deterministic = False, distengled = False , ndim_w = None  ):
        super().__init__()
        s0 = self.s0 = 7  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 1024  # nfilter_max
        size = 28
        ndim_u = latent_dim
        ndim_w = ndim_w
        self.distengled = distengled
        self.deterministic = deterministic
        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        if distengled == True:
            blocks_w = [
                ResnetBlock(nf, nf)
            ]
        
        blocks_u = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            if distengled == True:
                blocks_w += [
                    nn.AvgPool2d(3, stride=2, padding=1),
                    ResnetBlock(nf0, nf1),
                ]
            blocks_u += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]
        if distengled == True:
            self.conv_img_w = nn.Conv2d(3, 1*nf, 3, padding=1)
            self.resnet_w = nn.Sequential(*blocks_w)
            self.fc_mu_w = nn.Linear(self.nf0*s0*s0, ndim_w)
            self.fc_lv_w = nn.Linear(self.nf0*s0*s0, ndim_w)
        
        self.conv_img_u = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet_u = nn.Sequential(*blocks_u)
        self.fc_mu_u = nn.Linear(self.nf0 * s0 * s0, ndim_u)
        if deterministic == False:
            self.fc_lv_u = nn.Linear(self.nf0 * s0 * s0, ndim_u)


    def forward(self, x):
        # batch_size = x.size(0)
        if self.distengled:
            out_w = self.conv_img_w(x)
            out_w = self.resnet_w(out_w)
            out_w = out_w.view(out_w.size()[0], self.nf0*self.s0*self.s0)
            lv_w = self.fc_lv_w(out_w)

        out_u = self.conv_img_u(x)
        out_u = self.resnet_u(out_u)
        out_u = out_u.view(out_u.size()[0], self.nf0 * self.s0 * self.s0)


        

        if self.deterministic :
            return self.fc_mu_u(out_u)
        else:
            lv_u = self.fc_lv_u(out_u)
            if self.distengled:
                return self.fc_mu_w(out_w), F.softmax(lv_w, dim=-1) * lv_w.size(-1) + eta,\
                    self.fc_mu_u(out_u) , F.softmax(lv_u, dim=-1) * lv_u.size(-1) + eta
            else:
                return self.fc_mu_u(out_u) , F.softmax(lv_u, dim=-1) * lv_u.size(-1) + eta



class Dec(nn.Module):
    """ Generate a SVHN image given a sample from the latent space. """

    def __init__(self, latent_dim):
        super().__init__()

        # NOTE: I've set below variables according to Kieran's suggestions
        s0 = self.s0 = 7  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 512  # nfilter_max
        size = 28
        ndim = latent_dim
        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.fc = nn.Linear(ndim, self.nf0*s0*s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers-i), nf_max)
            nf1 = min(nf * 2**(nlayers-i-1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z):
        out = self.fc(z).view(-1, self.nf0, self.s0, self.s0)
        out = self.resnet(out)
        out = self.conv_img(actvn(out))

        if len(z.size()) == 2:
            out = out.view(*z.size()[:1], *out.size()[1:])
        else:
            out = out.view(*z.size()[:2], *out.size()[1:])

        # consider also predicting the length scale
        return out # mean, length scale

