
import numpy as np
import torch
import torch.nn as nn



class encoder_att(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.size_z = latent_dim
        self.enc_net = nn.Sequential(
            nn.Linear(18, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.mu_lin = nn.Linear(512, self.size_z)
        # self.logvar_lin = nn.Linear(512, self.size_z)

    def forward(self, x):
            return self.mu_lin(self.enc_net(x.float()))


class decoder_att(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        #  decoder network
        self.size_z = latent_dim
        self.dec_net = nn.Sequential(
            nn.Linear(self.size_z, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,18),
        )

    def forward(self, z):
            return self.dec_net(z)










class ResEncoderN(nn.Module):
    def __init__(self, channel_list, size_in=128, size_z=64, img_ch=3):
        super().__init__()
        self.img_ch = img_ch
        self.channel_list = channel_list
        self.size_z = size_z
        self.ch_enc = nn.Sequential(
            nn.Conv2d(self.img_ch, self.channel_list[0][0], 5, 1, 2),
            nn.BatchNorm2d(self.channel_list[0][0]),
            nn.LeakyReLU(0.1),
            nn.AvgPool2d(2),
        ) 

        self.size_in = size_in
        init_size = self.size_in // 2
        for i in self.channel_list:
            init_size = init_size // i[3]
        self.size_z_lin = (init_size * init_size) * (self.channel_list[-1][2] // 2)

        self.r_blocks = nn.ModuleList([RBlockN(*i) for i in self.channel_list])
        self.mu_lin = nn.Linear(self.size_z_lin, self.size_z)
        self.logvar_lin = nn.Linear(self.size_z_lin, self.size_z)
    
    def forward(self, x):
        x = self.ch_enc(x)
        for r_block in self.r_blocks:
            x = r_block(x)

        mu, _ = x.chunk(2, dim=1)
        mu = self.mu_lin(mu.view(mu.shape[0], -1))
        #logvar = self.logvar_lin(logvar.view(logvar.shape[0],-1))
        return mu










class RBlockN(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, up_rate=None, residual=True):
        super().__init__()
        self.down_rate = down_rate
        self.up_rate = up_rate
        self.residual = residual
        self.in_width = in_width
        self.middle_width = middle_width
        self.out_width = out_width
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_width,self.middle_width,3,1,1,bias=False),
            nn.BatchNorm2d(self.middle_width),
            nn.GELU(),
            nn.Conv2d(self.middle_width,self.out_width,3,1,1,bias=False),
            nn.BatchNorm2d(self.out_width),
        )
        self.sf = nn.GELU()
        self.size_conv = nn.Conv2d(self.in_width, self.out_width,1,1,0,bias=False)
        self.down_pool = nn.AvgPool2d(self.down_rate)
        self.up_pool = torch.nn.Upsample(scale_factor=self.up_rate, mode='bilinear')

    def forward(self, x):
        xhat = self.conv(x)
        if self.in_width != self.out_width:
            x = self.size_conv(x)
        xhat = self.sf(x + xhat)
        if self.down_rate is not None:
            xhat = self.down_pool(xhat)
        if self.up_rate is not None:
            xhat = self.up_pool(xhat)
        return xhat



class RBlock(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, up_rate=None, residual=True):
        super().__init__()
        self.down_rate = down_rate
        self.up_rate = up_rate
        self.residual = residual
        self.in_width = in_width
        self.middle_width = middle_width
        self.out_width = out_width
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_width,self.middle_width,3,1,1,bias=False),
            nn.BatchNorm2d(self.middle_width),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.middle_width,self.out_width,3,1,1,bias=False),
            nn.BatchNorm2d(self.out_width),
        )
        self.sf = nn.LeakyReLU(0.2)
        self.size_conv = nn.Conv2d(self.in_width, self.out_width,1,1,0,bias=False)
        self.down_pool = nn.AvgPool2d(self.down_rate)
        self.up_pool = torch.nn.Upsample(scale_factor=self.up_rate)

    def forward(self, x):
        xhat = self.conv(x)
        if self.in_width != self.out_width:
            x = self.size_conv(x)
        xhat = self.sf(x + xhat)
        if self.down_rate is not None:
            xhat = self.down_pool(xhat)
        if self.up_rate is not None:
            xhat = self.up_pool(xhat)
        return xhat





class Image_enc(nn.Module):
    def __init__(self, channel_list, size_in=128, latent_dim=64, img_ch=3):
        super().__init__()
        self.img_ch = img_ch
        self.channel_list = channel_list
        self.size_z = latent_dim
        self.ch_enc = nn.Sequential(
            nn.Conv2d(self.img_ch, self.channel_list[0][0], 5, 1, 2),
            nn.BatchNorm2d(self.channel_list[0][0]),
            nn.LeakyReLU(0.1),
            nn.AvgPool2d(2),
        ) 

        self.size_in = size_in
        init_size = self.size_in // 2
        for i in self.channel_list:
            init_size = init_size // i[3]
        self.size_z_lin = (init_size * init_size) * (self.channel_list[-1][2] // 2)

        self.r_blocks = nn.ModuleList([RBlockN(*i) for i in self.channel_list])
        self.mu_lin = nn.Linear(self.size_z_lin, self.size_z)
        self.logvar_lin = nn.Linear(self.size_z_lin, self.size_z)
    
    def forward(self, x):
        x = self.ch_enc(x)
        for r_block in self.r_blocks:
            x = r_block(x)
        mu, logvar = x.chunk(2, dim=1)
        mu = self.mu_lin(mu.view(mu.shape[0], -1))
        #logvar = self.logvar_lin(logvar.view(logvar.shape[0],-1))
        return mu #, logvar

class Image_dec(nn.Module):
    def __init__(self, channel_list, size_in=128, latent_dim=64, img_ch=3 ,enc_channel_list = None):
        super().__init__()
        self.img_ch = img_ch
        self.channel_list = channel_list
        self.size_z = latent_dim
        self.r_blocks = nn.ModuleList([RBlockN(i[0],i[1],i[2],None,i[3],True) for i in self.channel_list])
        self.ch_dec = nn.Sequential(
            RBlock(self.channel_list[-1][2], self.channel_list[-1][2], self.channel_list[-1][2]),
            nn.Conv2d(self.channel_list[-1][2], self.img_ch, 5, 1, 2),
            nn.Sigmoid()
        )

        init_size = size_in
        self.enc_channel_list =enc_channel_list
        for i in self.enc_channel_list:
            init_size = init_size // i[3]
        self.size_z_lin = (init_size * init_size) * (self.enc_channel_list[-1][2])

        self.z_lin = nn.Linear(self.size_z, self.size_z_lin)
        self.z_lin_relu = nn.ReLU()
        self.z_reshape_size = (self.size_z_lin // self.enc_channel_list[-1][2] // init_size)


    def forward(self, z):
        z = self.z_lin_relu(self.z_lin(z))
        x = z.view(z.shape[0],self.enc_channel_list[-1][2],self.z_reshape_size,self.z_reshape_size ) 
        for r_block in self.r_blocks:
            x = r_block(x)
        x = self.ch_dec(x)
        return x
