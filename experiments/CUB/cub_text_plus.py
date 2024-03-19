import os
import json

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader

from datasets_compatible import CUBSentences
from utils import Constants, FakeCategorical
from .vae import VAE

# Constants
maxSentLen = 32  # max length of any description for birds dataset
minOccur = 3
embeddingDim = 128
lenWindow = 3
fBase = 32
vocabSize = 1590
vocab_path = '../data/cub/oc:{}_msl:{}/cub.vocab'.format(minOccur, maxSentLen)

# Classes
class Enc(nn.Module):
    """ Generate latent parameters for sentence data. """

    def __init__(self, latentDim_w, latentDim_u):
        super(Enc, self).__init__()
        self.embedding = nn.Linear(vocabSize, embeddingDim)
        self.enc_w = nn.Sequential(
            # input size: 1 x 32 x 128
            nn.Conv2d(1, fBase, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 16 x 64
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 32
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True)
        )
        self.enc_u = nn.Sequential(
            # input size: 1 x 32 x 128
            nn.Conv2d(1, fBase, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 16 x 64
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 32
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # # size: (fBase * 4) x 4 x 16
            nn.Conv2d(fBase * 4, fBase * 8, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.Conv2d(fBase * 8, fBase * 16, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
        )
        self.c1_w = nn.Linear(fBase * 16 * 16, latentDim_w)
        self.c2_w = nn.Linear(fBase * 16 * 16, latentDim_w)

        self.c1_u = nn.Conv2d(fBase * 16, latentDim_u, 4, 1, 0, bias=True)
        self.c2_u = nn.Conv2d(fBase * 16, latentDim_u, 4, 1, 0, bias=True)

    def forward(self, x):
        x_emb = self.embedding(x).unsqueeze(1)
        e_w = self.enc_w(x_emb)
        e_w = e_w.view(-1, fBase * 16 * 16)
        mu_w, lv_w = self.c1_w(e_w), self.c2_w(e_w)
        e_u = self.enc_u(x_emb)
        mu_u, lv_u = self.c1_u(e_u).squeeze(), self.c2_u(e_u).squeeze()
        # mu_u, lv_u = self.c1_u(e_u).squeeze().unsqueeze(0), self.c2_u(e_u).squeeze().unsqueeze(0)
        return torch.cat((mu_w, mu_u), dim=1), \
               torch.cat((F.softplus(lv_w) + Constants.eta,
                          F.softplus(lv_u) + Constants.eta), dim=1)


class Dec(nn.Module):
    """ Generate a sentence given a sample from the latent space. """

    def __init__(self, latentDim_w, latentDim_u):
        super(Dec, self).__init__()
        self.dec_w = nn.Sequential(
            nn.ConvTranspose2d(latentDim_w, fBase * 16, 4, 1, 0, bias=True),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
            nn.ConvTranspose2d(fBase * 16, fBase * 8, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.ConvTranspose2d(fBase * 8, fBase * 4, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
        )
        self.dec_u = nn.Sequential(
            nn.ConvTranspose2d(latentDim_u, fBase * 16, 4, 1, 0, bias=True),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
            nn.ConvTranspose2d(fBase * 16, fBase * 8, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
            nn.ConvTranspose2d(fBase * 8, fBase * 8, 3, 1, 1, bias=True),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.ConvTranspose2d(fBase * 8, fBase * 4, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
        )
        self.dec_h = nn.Sequential(
            nn.ConvTranspose2d(fBase * 8, fBase * 4, 3, 1, 1, bias=True),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 32
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 64
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 32 x 128
            nn.ConvTranspose2d(fBase, 1, 4, 2, 1, bias=True),
            nn.ReLU(True)
            # Output size: 1 x 64 x 256
        )
        # inverts the 'embedding' module upto one-hotness
        self.toVocabSize = nn.Linear(embeddingDim, vocabSize)

        self.latent_dim_w = latentDim_w
        self.latent_dim_u = latentDim_u

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z):
        #z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        w, u = torch.split(z, [self.latent_dim_w, self.latent_dim_u], dim=-1)
        u = u.unsqueeze(-1).unsqueeze(-1)
        hu = self.dec_u(u.view(-1, *u.size()[-3:]))
        w = w.unsqueeze(-1).unsqueeze(-1)
        hw = self.dec_w(w.view(-1, *w.size()[-3:]))
        h = torch.cat((hw, hu), dim=1)
        out = self.dec_h(h)
        out = out.view(*u.size()[:-3], *out.size()[1:]).view(-1, embeddingDim)
        # The softmax is key for this to work
        ret = [self.softmax(self.toVocabSize(out).view(*u.size()[:-3], maxSentLen, vocabSize))]
        return ret