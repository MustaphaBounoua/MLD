from audioop import bias
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Bernoulli
import numpy as np
import torch.nn.functional as F

KERNEL_SIZE = 4
STRIDE = 2
PADDING = 1
eta=1e-6
image_channels = 1
image_side = 28
image_conv_layers = [32, 64]
image_linear_layers = [128, 128]
image_mod_latent_dim = 64
image_input_dim = 28*28


class ImageEncoder(nn.Module):
    def __init__(self,latent_dim  , name="", input_dim = image_input_dim, 
                 n_channels = image_channels, 
                 conv_layers =image_conv_layers, 
                 linear_layers =image_linear_layers,
                 deterministic = False
                 ):
        
        super(ImageEncoder, self).__init__()

        # Variables
        self.name = name
        self.input_dim = input_dim
        self.n_channels = n_channels
        self.conv_layers = conv_layers
        self.linear_layers = linear_layers
        self.output_dim = latent_dim
        self.deterministic = deterministic
        # Create Network
        pre = n_channels
        conv_output_side = input_dim
        self.features = []
        image_conv_layers = [32, 64]
        for ls in conv_layers:
            pos = ls
            self.features.append(
                nn.Conv2d(pre, pos, KERNEL_SIZE, STRIDE, PADDING, bias=False))
            self.features.append(Swish())
            conv_output_side = convolutional_output_width(
                conv_output_side, KERNEL_SIZE, PADDING, STRIDE)
            pre = pos
        self.features = nn.Sequential(*self.features)
        pos_conv_n_channels = conv_layers[-1]

        # Size of the unrolled images at the end of features
        # pre = pos_conv_n_channels * conv_output_side * conv_output_side
        pre = 64 * 7 * 7                                                                        # Fix this
        self.classifier = []
        for ls in linear_layers:
            pos = ls
            self.classifier.append(nn.Linear(pre, pos))
            self.classifier.append(Swish())
            pre = pos
   
        # Output layer of the network
        self.fc_mu = nn.Linear(pre, self.output_dim )
        self.fc_logvar = nn.Linear(pre, self.output_dim )


        self.classifier = nn.Sequential(*self.classifier)

    def forward(self, x):

        x = self.features(x)
        h = x.view(x.size(0), -1)
        h = self.classifier(h)
        
        if self.deterministic:
            return self.fc_mu(h)
        else:
            return self.fc_mu(h), self.fc_logvar(h)






class MHDImageEncoderPLus(nn.Module):
    def __init__(self,latent_dim  ,latent_dim_w, name="", input_dim = image_input_dim, 
                 n_channels = image_channels, 
                 conv_layers =image_conv_layers, 
                 linear_layers =image_linear_layers,
                 deterministic = False
                 ):
        
        super(MHDImageEncoderPLus, self).__init__()

        # Variables
        self.name = name
        self.latent_dim =latent_dim
        self.input_dim = input_dim
        self.n_channels = n_channels
        self.conv_layers = conv_layers
        self.linear_layers = linear_layers
        self.output_dim = latent_dim
        self.deterministic = deterministic
        # Create Network
        pre = n_channels
        conv_output_side = input_dim
        self.features = []
        image_conv_layers = [32, 64]
        for ls in conv_layers:
            pos = ls
            self.features.append(
                nn.Conv2d(pre, pos, KERNEL_SIZE, STRIDE, PADDING, bias=False))
            self.features.append(Swish())
            conv_output_side = convolutional_output_width(
                conv_output_side, KERNEL_SIZE, PADDING, STRIDE)
            pre = pos
        self.features = nn.Sequential(*self.features)
        pos_conv_n_channels = conv_layers[-1]

        # Size of the unrolled images at the end of features
        # pre = pos_conv_n_channels * conv_output_side * conv_output_side
        pre = 64 * 7 * 7   
        
        self.classifier = [] 
        self.classifier_w = []

        for ls in linear_layers:
            pos = ls
            self.classifier.append(nn.Linear(pre, pos))
            self.classifier.append(Swish())
            pre = pos

        for ls in linear_layers:
            pos = ls
            self.classifier_w.append(nn.Linear(pre, pos))
            self.classifier_w.append(Swish())
            pre = pos

        self.fc_mu_w= nn.Linear(pre, latent_dim_w )
        self.fc_logvar_w = nn.Linear(pre, latent_dim_w )
        # Output layer of the network
        self.fc_mu = nn.Linear(pre, self.output_dim )
        self.fc_logvar = nn.Linear(pre, self.output_dim )

        self.classifier_w = nn.Sequential(*self.classifier_w)
        self.classifier = nn.Sequential(*self.classifier)

    def forward(self, x):

        x = self.features(x)
        h = x.view(x.size(0), -1)
        
        h = self.classifier(h)
        h_w=self.classifier_w(h)

        latent_space_mu=self.fc_mu(h)
        latent_space_logvar =self.fc_logvar(h)

        mu_u  = latent_space_mu
        l_u =latent_space_logvar

        mu_w = self.fc_mu_w(h_w)
        l_w = self.fc_logvar_w(h_w)

        return mu_w, F.softmax(l_w, dim=-1) * l_w.size(-1) + eta, mu_u, F.softmax(l_u, dim=-1) * l_u.size(-1) + eta


  


# Image Decoder
class ImageDecoder(nn.Module):
    def __init__(self,latent_dim, name="", n_channels = image_channels, 
                 conv_layers = np.flip(image_conv_layers),
                 linear_layers =image_linear_layers, 
                 output_dim = image_mod_latent_dim ):
        super(ImageDecoder, self).__init__()

        # Variables
        self.name = name
        self.input_dim = latent_dim
        self.n_channels = n_channels
        self.conv_layers = conv_layers
        self.linear_layers = linear_layers
        self.output_dim = output_dim

        # Create Network
        self.conv_output_side = 7
        self.pos_conv_n_channels = 64
        pos_conv_size = self.pos_conv_n_channels * (self.conv_output_side ** 2)

        self.upsampler = []
        pre = latent_dim
        mod_linear_layers_sizes = list(linear_layers) + [pos_conv_size]

        for ls in mod_linear_layers_sizes:
            pos = ls
            self.upsampler.append(nn.Linear(pre, pos))
            self.upsampler.append(Swish())
            pre = pos
        self.upsampler = nn.Sequential(*self.upsampler)

        self.hallucinate = []
        pre = conv_layers[0]
        for ls in conv_layers[1:]:
            pos = ls
            self.hallucinate.append(
                nn.ConvTranspose2d(
                    pre, pos, KERNEL_SIZE, STRIDE, PADDING, bias=False))
            self.hallucinate.append(Swish())
            pre = pos
        self.hallucinate.append(
            nn.ConvTranspose2d(
                pre, n_channels, KERNEL_SIZE, STRIDE, PADDING, bias=False))
        self.hallucinate = nn.Sequential(*self.hallucinate)


        # Output Transformation
        self.out_process = nn.Sigmoid()



    def forward(self, x):
        x = self.upsampler(x)
        x = x.view(-1, self.pos_conv_n_channels, self.conv_output_side,
                   self.conv_output_side)
        out = self.hallucinate(x)
        return self.out_process(out)








class ImageDecoderPlus(nn.Module):
    def __init__(self,latent_dim, name="", n_channels = image_channels, 
                 conv_layers = np.flip(image_conv_layers),
                 linear_layers =image_linear_layers, 
                 output_dim = image_mod_latent_dim ):
        super(ImageDecoderPlus, self).__init__()

        # Variables
        self.name = name
        self.input_dim = latent_dim
        self.n_channels = n_channels
        self.conv_layers = conv_layers
        self.linear_layers = linear_layers
        self.output_dim = output_dim

        # Create Network
        self.conv_output_side = 7
        self.pos_conv_n_channels = 64
        pos_conv_size = self.pos_conv_n_channels * (self.conv_output_side ** 2)

        self.upsampler = []
        pre = latent_dim
        mod_linear_layers_sizes = list(linear_layers) + [pos_conv_size]

        for ls in mod_linear_layers_sizes:
            pos = ls
            self.upsampler.append(nn.Linear(pre, pos))
            self.upsampler.append(Swish())
            pre = pos
        self.upsampler = nn.Sequential(*self.upsampler)

        self.hallucinate = []
        pre = conv_layers[0]
        for ls in conv_layers[1:]:
            pos = ls
            self.hallucinate.append(
                nn.ConvTranspose2d(
                    pre, pos, KERNEL_SIZE, STRIDE, PADDING, bias=False))
            self.hallucinate.append(Swish())
            pre = pos
        self.hallucinate.append(
            nn.ConvTranspose2d(
                pre, n_channels, KERNEL_SIZE, STRIDE, PADDING, bias=False))
        self.hallucinate = nn.Sequential(*self.hallucinate)


        # Output Transformation
        self.out_process = nn.Sigmoid()



    def forward(self, z):
        x = self.upsampler(z)

        x = x.view(-1, self.pos_conv_n_channels, self.conv_output_side,
                   self.conv_output_side)
        out = self.hallucinate(x)


        if len(z.size()) == 2:
            out = out.view(*z.size()[:1], *out.size()[1:])
        else:
            out = out.view(*z.size()[:2], *out.size()[1:])
            
        return self.out_process(out)







# Extra Components
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def convolutional_output_width(input_width, kernel_width, padding, stride):
    # assumes square input/output and kernels
    return int((input_width - kernel_width + 2 * padding) / stride + 1)

def sequence_convolutional_output_width(input_width, conv_layers_sizes,
                                        kernel_width, padding, stride):
    # assumes square input/output and kernels
    output_width = input_width
    for ls in conv_layers_sizes:
        output_width = convolutional_output_width(output_width, kernel_width,
                                                  padding, stride)

    return output_width