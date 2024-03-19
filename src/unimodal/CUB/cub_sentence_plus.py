# Sentence model specification - real CUB image version

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


# Constants
maxSentLen = 32  # max length of any description for birds dataset
minOccur = 3
embeddingDim = 128
lenWindow = 3
fBase = 32
vocabSize = 1590
eta = 1e-6
# Classes
class Enc(nn.Module):
    """ Generate latent parameters for sentence data. """

    def __init__(self, latent_dim,deterministic = False, distengeled = False , latent_dim_w = None  ):
        super(Enc, self).__init__()
        self.embedding = nn.Linear(vocabSize, embeddingDim)
        self.deterministic =deterministic
        self.distengeled =distengeled
        if self.distengeled:
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
        if self.distengeled:
            self.c1_w = nn.Linear(fBase * 16 * 16, latent_dim_w)
            self.c2_w = nn.Linear(fBase * 16 * 16, latent_dim_w)

        self.c1_u = nn.Conv2d(fBase * 16, latent_dim, 4, 1, 0, bias=True)

        if self.deterministic == False:
            self.c2_u = nn.Conv2d(fBase * 16, latent_dim, 4, 1, 0, bias=True)

    def forward(self, x):
        x_emb = self.embedding(x).unsqueeze(1)
        if self.distengeled:
            e_w = self.enc_w(x_emb)
            e_w = e_w.view(-1, fBase * 16 * 16)
        
        e_u = self.enc_u(x_emb)


        mu_u  = self.c1_u(e_u).squeeze()
        
        # mu_u, lv_u = self.c1_u(e_u).squeeze().unsqueeze(0), self.c2_u(e_u).squeeze().unsqueeze(0)


        if self.deterministic :
            return mu_u
        else:
            lv_u = self.c2_u(e_u).squeeze()
            if self.distengeled:
                
                mu_w, lv_w = self.c1_w(e_w), self.c2_w(e_w)

                return mu_w, F.softmax(lv_w, dim=-1) * lv_w.size(-1) + 1e-6, \
                    mu_u , F.softmax(lv_u, dim=-1) * lv_u.size(-1) + 1e-6
            else:
                return mu_u , F.softplus(lv_u) + eta

        


class Dec(nn.Module):
    """ Generate a sentence given a sample from the latent space. """

    def __init__(self, latent_dim,latent_dim_w=None, distengeled=False):
        
        super(Dec, self).__init__()

        self.distengeled =distengeled
        if self.distengeled:
            self.dec_w = nn.Sequential(
                nn.ConvTranspose2d(latent_dim_w, fBase * 16, 4, 1, 0, bias=True),
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
            nn.ConvTranspose2d(latent_dim, fBase * 16, 4, 1, 0, bias=True),
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
        if self.distengeled:
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
        else:
            self.dec_h = nn.Sequential(
            nn.ConvTranspose2d(fBase * 4, fBase * 4, 3, 1, 1, bias=True),
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

        self.latent_dim_w = latent_dim_w
        self.latent_dim_u = latent_dim

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z):
        
        #z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        if self.distengeled:
            w, u = torch.split(z, [self.latent_dim_w, self.latent_dim_u], dim=-1)
            u = u.unsqueeze(-1).unsqueeze(-1)
            hu = self.dec_u(u.view(-1, *u.size()[-3:]))
            w = w.unsqueeze(-1).unsqueeze(-1)
            hw = self.dec_w(w.view(-1, *w.size()[-3:]))
            h = torch.cat((hw, hu), dim=1)
            out = self.dec_h(h)

           

            out = out.view(*u.size()[:-3], *out.size()[1:]).view(-1,z.size(0), embeddingDim)
            # The softmax is key for this to work
            

            ret = self.softmax(self.toVocabSize(out).view(*u.size()[:-3], maxSentLen, vocabSize))
            
            return ret
        else:
            
            u = z.unsqueeze(-1).unsqueeze(-1)
            hu = self.dec_u(u.view(-1, *u.size()[-3:]))

            hu = self.dec_h(hu)
           # print("hu.shape")
           # print(hu.shape)
            out = hu.view(*u.size()[:-3], *hu.size()[1:]).view(-1,z.size(0), embeddingDim)
           # print("out.shape")
          #  print(out.shape)
            output = self.toVocabSize(out)
          #  print("output.shape")
          #  print(output.shape)     
            output=output.view(*u.size()[:-3], maxSentLen, vocabSize)                 
            ret = self.softmax(output)
            return ret


if __name__ =="__main__":
    img = torch.randn(64,32,1590)
    enc = Enc(latent_dim=64,latent_dim_w=32,distengeled=False,deterministic=True)
    dec = Dec(latent_dim=64,latent_dim_w=32,distengeled=False)
    print(enc)
    print(dec)
    z = enc(img)
    print(z.shape)
    y = dec(torch.randn(64,64))
    print(y.shape)