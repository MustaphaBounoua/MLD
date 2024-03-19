import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock1dTransposeConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, o_padding, upsample, a=2,
                 b=0.3):
        super(ResidualBlock1dTransposeConv, self).__init__()
        self.bn1 = nn.BatchNorm1d(channels_in)
        self.conv1 = nn.ConvTranspose1d(channels_in, channels_in, kernel_size=1, stride=1, padding=0)
        self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(channels_in)
        self.conv2 = nn.ConvTranspose1d(channels_in, channels_out, kernel_size=kernelsize, stride=stride,
                                        padding=padding, dilation=dilation, output_padding=o_padding)
        self.dropout2 = nn.Dropout(p=0.5, inplace=False)
        self.upsample = upsample
        self.a = a
        self.b = b

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        if self.upsample:
            residual = self.upsample(x)
        out = self.a * residual + self.b * out
        return out


class DecoderText(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
   
        dim_text = 32
        num_features = 1590
        z_dim = latent_dim

        self.fc = nn.Linear(z_dim, 5 * dim_text)
        self.resblock_1 = DecoderText.res_block_decoder(5 * dim_text, 5 * dim_text,
                                                      kernelsize=4, stride=1, padding=0, dilation=1, o_padding=0)
        self.resblock_2 = DecoderText.res_block_decoder(5 * dim_text, 5 * dim_text,
                                                      kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.resblock_3 = DecoderText.res_block_decoder(5 * dim_text, 4 * dim_text,
                                                      kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.resblock_4 = DecoderText.res_block_decoder(4 * dim_text, 3 * dim_text,
                                                      kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.resblock_5 = DecoderText.res_block_decoder(3 * dim_text, 2 * dim_text,
                                                      kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.resblock_6 = DecoderText.res_block_decoder(2 * dim_text, dim_text,
                                                      kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.conv2 = nn.ConvTranspose1d(dim_text, num_features,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        dilation=1,
                                        output_padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        
        d = self.fc(z)
        d = d.unsqueeze(-1)
        d = self.resblock_3(d)
        d = self.resblock_4(d)
        d = self.resblock_5(d)
        d = self.resblock_6(d)
        d = self.conv2(d)
        d = self.softmax(d)
        d = d.transpose(-2,-1)
        return d

    @staticmethod
    def res_block_decoder(in_channels, out_channels, kernelsize, stride, padding, o_padding, dilation, a_val=2.0,
                          b_val=0.3):
        upsample = None

        if (kernelsize != 1 or stride != 1) or (in_channels != out_channels) or dilation != 1:
            upsample = nn.Sequential(nn.ConvTranspose1d(in_channels, out_channels,
                                                        kernel_size=kernelsize,
                                                        stride=stride,
                                                        padding=padding,
                                                        dilation=dilation,
                                                        output_padding=o_padding),
                                     nn.BatchNorm1d(out_channels))
        layers = []
        layers.append(
            ResidualBlock1dTransposeConv(in_channels, out_channels, kernelsize, stride, padding, dilation, o_padding,
                                         upsample=upsample, a=a_val, b=b_val))
        return nn.Sequential(*layers)


class ResidualBlock1dConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, downsample, a=2, b=0.3):
        super(ResidualBlock1dConv, self).__init__()
        self.bn1 = nn.BatchNorm1d(channels_in)
        self.conv1 = nn.Conv1d(channels_in, channels_in, kernel_size=1, stride=1, padding=0)
        self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(channels_in)
        self.conv2 = nn.Conv1d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding,
                               dilation=dilation)
        self.dropout2 = nn.Dropout(p=0.5, inplace=False)
        self.downsample = downsample
        self.a = a
        self.b = b

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        if self.downsample:
            residual = self.downsample(x)
        out = self.a * residual + self.b * out
        return out


class EncoderText(nn.Module):
    def __init__(self, latent_dim,deterministic=False, sigmoid=False):
        super().__init__()
        self.deterministic =deterministic
        self.sigmoid = sigmoid
        dim_text = 32
        num_features = 1590

        self.conv1 = nn.Conv1d(num_features, dim_text,
                               kernel_size=4, stride=2, padding=1, dilation=1)
        self.resblock_1 = EncoderText.make_res_block_encoder_feature_extractor(dim_text,
                                                                                 2 * dim_text,
                                                                                 kernelsize=4, stride=2, padding=1,
                                                                                 dilation=1)
        self.resblock_2 = EncoderText.make_res_block_encoder_feature_extractor(2 * dim_text,
                                                                                 3 * dim_text,
                                                                                 kernelsize=4, stride=2, padding=1,
                                                                                 dilation=1)
        self.resblock_3 = EncoderText.make_res_block_encoder_feature_extractor(3 * dim_text,
                                                                                 4 * dim_text,
                                                                                 kernelsize=4, stride=2, padding=1,
                                                                                 dilation=1)
        self.resblock_4 = EncoderText.make_res_block_encoder_feature_extractor(4 * dim_text,
                                                                                 5 * dim_text,
                                                                                 kernelsize=4, stride=2, padding=1,
                                                                                 dilation=1)
        self.resblock_5 = EncoderText.make_res_block_encoder_feature_extractor(5 * dim_text,
                                                                                 5 * dim_text,
                                                                                 kernelsize=4, stride=2, padding=1,
                                                                                 dilation=1)
        self.resblock_6 = EncoderText.make_res_block_encoder_feature_extractor(5 * dim_text,
                                                                                 5 * dim_text,
                                                                                 kernelsize=4, stride=2, padding=0,
                                                                                 dilation=1)
        self.fc_mu = nn.Linear(5 * dim_text, latent_dim)
        self.fc_logvar = nn.Linear(5 * dim_text, latent_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.transpose(-2, -1)
        out = self.conv1(x)
        out = self.resblock_1(out)
        out = self.resblock_2(out)
        out = self.resblock_3(out)
        out = self.resblock_4(out)
        out = out.view(batch_size, -1)
        out = self.relu(out)
        
        if self.deterministic:
            return self.fc_mu(out)
        else:
            return self.fc_mu(out) , self.fc_logvar(out)
         

    @staticmethod
    def make_res_block_encoder_feature_extractor(in_channels, out_channels, kernelsize, stride, padding, dilation,
                                                 a_val=2.0, b_val=0.3):
        downsample = None
        if (stride != 1) or (in_channels != out_channels) or dilation != 1:
            downsample = nn.Sequential(nn.Conv1d(in_channels, out_channels,
                                                 kernel_size=kernelsize,
                                                 stride=stride,
                                                 padding=padding,
                                                 dilation=dilation),
                                       nn.BatchNorm1d(out_channels))
        layers = []
        layers.append(
            ResidualBlock1dConv(in_channels, out_channels, kernelsize, stride, padding, dilation, downsample, a=a_val,
                                b=b_val))
        return nn.Sequential(*layers)
