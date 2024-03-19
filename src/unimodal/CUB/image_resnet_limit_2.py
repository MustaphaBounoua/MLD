import torch
import torch.nn as nn
import numpy as np


num_layers_img= 5
DIM_img =96

class EncoderImg(nn.Module):
    def __init__(self, latent_dim = 128,deterministic =False):
        super(EncoderImg, self).__init__();
        self.deterministic=deterministic
        self.feature_extractor = FeatureExtractorImg( a=2.0, b=0.3)
        self.mu = nn.Linear(num_layers_img * DIM_img, latent_dim)
        self.sigma = nn.Linear(num_layers_img*DIM_img,latent_dim)
                                                          

    def forward(self, x_img):
        h_img = self.feature_extractor(x_img);
        h_img = h_img.view(h_img.shape[0], h_img.shape[1]*h_img.shape[2])
        if self.deterministic:
            return self.mu(h_img)
        else:
            return self.mu(h_img), self.sigma(h_img)
     


class DecoderImg(nn.Module):
    def __init__(self, latent_dim = 128):
        super(DecoderImg, self).__init__();

        self.lin = nn.Linear(latent_dim,num_layers_img*DIM_img)
        self.img_generator = DataGeneratorImg( a=2.0, b=0.3)
        self.sigmoid = nn.Sigmoid()        
                                                          

    def forward(self, x_img):
        h_img = self.lin(x_img);
        h_img=h_img.view(h_img.size(0), h_img.size(1), 1, 1)
        #return self.img_generator(h_img) 
        return self.sigmoid (self.img_generator(h_img) )















def res_block_gen(in_channels, out_channels, kernelsize, stride, padding, o_padding, dilation, a_val=1.0, b_val=1.0):
    upsample = None;
    if (kernelsize != 1 and stride != 1) or (in_channels != out_channels):
        upsample = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels,
                                                    kernel_size=kernelsize,
                                                    stride=stride,
                                                    padding=padding,
                                                    dilation=dilation,
                                                    output_padding=o_padding),
                                 nn.BatchNorm2d(out_channels))
    layers = [];
    layers.append(ResidualBlock2dTransposeConv(in_channels, out_channels,
                                               kernelsize=kernelsize,
                                               stride=stride,
                                               padding=padding,
                                               dilation=dilation,
                                               o_padding=o_padding,
                                               upsample=upsample,
                                               a=a_val, b=b_val))
    return nn.Sequential(*layers)


class DataGeneratorImg(nn.Module):
    def __init__(self, a, b):
        super(DataGeneratorImg, self).__init__()
    
        self.a = a;
        self.b = b;
        self.resblock1 = res_block_gen(5*DIM_img, 4*DIM_img, kernelsize=4, stride=1, padding=0, dilation=1, o_padding=0, a_val=a, b_val=b);
        self.resblock2 = res_block_gen(4*DIM_img, 3*DIM_img, kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b);
        self.resblock3 = res_block_gen(3*DIM_img, 2*DIM_img, kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b);
        self.resblock4 = res_block_gen(2*DIM_img, 1*DIM_img, kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b);
        self.conv = nn.ConvTranspose2d(DIM_img, 3,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       dilation=1,
                                       output_padding=1);

    def forward(self, feats):
        # d = self.data_generator(feats)
        d = self.resblock1(feats);
        d = self.resblock2(d);
        d = self.resblock3(d);
        d = self.resblock4(d);
        # d = self.resblock5(d);
        d = self.conv(d)
        return d;






def make_res_block_feature_extractor(in_channels, out_channels, kernelsize, stride, padding, dilation, a_val=2.0, b_val=0.3):
    downsample = None;
    if (stride != 2) or (in_channels != out_channels):
        downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                             kernel_size=kernelsize,
                                             padding=padding,
                                             stride=stride,
                                             dilation=dilation),
                                   nn.BatchNorm2d(out_channels))
    layers = [];
    layers.append(ResidualBlock2dConv(in_channels, out_channels, kernelsize, stride, padding, dilation, downsample,a=a_val, b=b_val))
    return nn.Sequential(*layers)


class FeatureExtractorImg(nn.Module):
    def __init__(self, a, b):
        super(FeatureExtractorImg, self).__init__();
        self.a = a;
        self.b = b;
        self.conv1 = nn.Conv2d(3, DIM_img,
                              kernel_size=3,
                              stride=2,
                              padding=2,
                              dilation=1,
                              bias=False)
        self.resblock1 = make_res_block_feature_extractor(DIM_img, 2 * DIM_img, kernelsize=4, stride=2,
                                                          padding=1, dilation=1, a_val=a, b_val=b)
        self.resblock2 = make_res_block_feature_extractor(2 * DIM_img, 3 * DIM_img, kernelsize=4, stride=2,
                                                          padding=1, dilation=1, a_val=self.a, b_val=self.b)
        self.resblock3 = make_res_block_feature_extractor(3 * DIM_img, 4 * DIM_img, kernelsize=4, stride=2,
                                                          padding=1, dilation=1, a_val=self.a, b_val=self.b)
        self.resblock4 = make_res_block_feature_extractor(4 * DIM_img, 5 * DIM_img, kernelsize=4, stride=2,
                                                          padding=0, dilation=1, a_val=self.a, b_val=self.b)

    def forward(self, x):

        out = self.conv1(x)
        out = self.resblock1(out);
        out = self.resblock2(out);
        out = self.resblock3(out);
        out = self.resblock4(out);
        return out










class ResidualBlock2dConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, downsample, a=1, b=1):
        super(ResidualBlock2dConv, self).__init__();
        self.conv1 = nn.Conv2d(channels_in, channels_in, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False)
        self.dropout1 = nn.Dropout2d(p=0.5, inplace=False)
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(channels_in)
        self.conv2 = nn.Conv2d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.dropout2 = nn.Dropout2d(p=0.5, inplace=False)
        self.downsample = downsample
        self.a = a;
        self.b = b;

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
     #   out = self.dropout1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
    #    out = self.dropout2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.a*residual + self.b*out;
        return out


class ResidualBlock2dTransposeConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, o_padding, upsample, a=1, b=1):
        super(ResidualBlock2dTransposeConv, self).__init__();
        self.conv1 = nn.ConvTranspose2d(channels_in, channels_in, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False)
        self.dropout1 = nn.Dropout2d(p=0.5, inplace=False);
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(channels_in)
        self.conv2 = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, dilation=dilation, bias=False, output_padding=o_padding)
        self.dropout2 = nn.Dropout2d(p=0.5, inplace=False);
        # self.conv3 = nn.ConvTranspose2d(channels_out, channels_out, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False)
        # self.bn3 = nn.BatchNorm2d(channels_out)
        self.upsample = upsample
        self.a = a;
        self.b = b;

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
     #   out = self.dropout1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
      #  out = self.dropout2(out)
        if self.upsample is not None:
            residual = self.upsample(x)
        out = self.a * residual + self.b * out;
        return out
    
    
    
if __name__ =="__main__":
    img = torch.randn(64,3,64,64)
    enc = EncoderImg(128,deterministic=True)
    dec = DecoderImg(128)
    print(enc)
    print(dec)
    z = enc(img)
    print(z.shape)
    y = dec(z)
    print(y.shape)
    
