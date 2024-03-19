import torch
import torch.nn as nn
from src.unimodal.mnist_svhn.MnistLabelVAE import FeatureEncText
import torch.nn.functional as F




class MnistClassifier(nn.Module):
    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2);
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2);
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2);
        self.relu = nn.ReLU();
        self.d = nn.d(p=0.5, inplace=False);
        self.linear = nn.Linear(in_features=128, out_features=10, bias=True)  # 10 is the number of classes (=digits)
        self.sigmoid = nn.Sigmoid();

    def forward(self, x):
        h = self.conv1(x);
        h = self.relu(h);
        h = self.conv2(h);
        h = self.relu(h);
        h = self.conv3(h);
        h = self.relu(h);
        h = self.d(h);
        h = h.view(h.size(0), -1);
        h = self.linear(h);
        out = self.sigmoid(h);
        return out;

    def get_activations(self, x):
        h = self.conv1(x)
        h = self.relu(h)
        h = self.conv2(h);
        h = self.relu(h);
        h = self.conv3(h);
        h = self.relu(h);
        h = self.d(h);
        h = h.view(h.size(0), -1);
        return h
    
    
    




class SVHNClassifier(nn.Module):
    def __init__(self):
        super(SVHNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, dilation=1);
        self.bn1 = nn.BatchNorm2d(32);
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, dilation=1);
        self.bn2 = nn.BatchNorm2d(64);
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, dilation=1);
        self.bn3 = nn.BatchNorm2d(64);
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0, dilation=1);
        self.bn4 = nn.BatchNorm2d(128);
        self.relu = nn.ReLU();
        self.d = nn.d(p=0.5, inplace=False);
        self.linear = nn.Linear(in_features=128, out_features=10, bias=True)  # 10 is the number of classes (=digits)
        self.sigmoid = nn.Sigmoid();

    def forward(self, x):
        h = self.conv1(x);
        h = self.d(h);
        h = self.bn1(h);
        h = self.relu(h);
        h = self.conv2(h);
        h = self.d(h);
        h = self.bn2(h);
        h = self.relu(h);
        h = self.conv3(h);
        h = self.d(h);
        h = self.bn3(h);
        h = self.relu(h);
        h = self.conv4(h);
        h = self.d(h);
        h = self.bn4(h);
        h = self.relu(h);
        h = h.view(h.size(0), -1);
        h = self.linear(h);
        out = self.sigmoid(h);
        return out;

    def get_activations(self, x):
        h = self.conv1(x);
        h = self.d(h);
        h = self.bn1(h);
        h = self.relu(h);
        h = self.conv2(h);
        h = self.d(h);
        h = self.bn2(h);
        h = self.relu(h);
        h = self.conv3(h);
        h = self.d(h);
        h = self.bn3(h);
        h = self.relu(h);
        h = self.conv4(h);
        h = self.d(h);
        h = self.bn4(h);
        h = self.relu(h);
        h = h.view(h.size(0), -1);
        return h;
    
    
    
    

# Residual block
class ResidualBlockEncoder(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, downsample):
        super(ResidualBlockEncoder, self).__init__()
        self.bn1 = nn.BatchNorm1d(channels_in);
        self.conv1 = nn.Conv1d(channels_in, channels_in, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(channels_in);
        self.conv2 = nn.Conv1d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, dilation=dilation)
        self.downsample = downsample;

    def forward(self, x):
        residual = x;
        out = self.bn1(x)
        out = self.relu(out);
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x);
        out = residual + 0.3*out
        return out



class ResidualBlockDecoder(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, upsample):
        super(ResidualBlockDecoder, self).__init__()
        self.bn1 = nn.BatchNorm1d(channels_in);
        self.conv1 = nn.ConvTranspose1d(channels_in, channels_in, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(channels_in);
        self.conv2 = nn.ConvTranspose1d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, dilation=dilation, output_padding=1)
        self.upsample = upsample;


    def forward(self, x):
        residual = x;
        out = self.bn1(x)
        out = self.relu(out);
        out = self.conv1(out);
        out = self.bn2(out);
        out = self.relu(out);
        out = self.conv2(out);
        if self.upsample:
            residual = self.upsample(x);
        out = 2.0*residual + 0.3*out
        return out


def make_res_block_encoder(channels_in, channels_out, kernelsize, stride, padding, dilation):
    downsample = None;
    if (stride != 1) or (channels_in != channels_out) or dilation != 1:
        downsample = nn.Sequential(nn.Conv1d(channels_in, channels_out,
                                             kernel_size=kernelsize,
                                             stride=stride,
                                             padding=padding,
                                             dilation=dilation),
                                   nn.BatchNorm1d(channels_out))
    layers = []
    layers.append(ResidualBlockEncoder(channels_in, channels_out, kernelsize, stride, padding, dilation, downsample))
    return nn.Sequential(*layers)


def make_res_block_decoder(channels_in, channels_out, kernelsize, stride, padding, dilation):
    upsample = None;
    if (kernelsize != 1 or stride != 1) or (channels_in != channels_out) or dilation != 1:
        upsample = nn.Sequential(nn.ConvTranspose1d(channels_in, channels_out,
                                                    kernel_size=kernelsize,
                                                    stride=stride,
                                                    padding=padding,
                                                    dilation=dilation,
                                                    output_padding=1),
                                   nn.BatchNorm1d(channels_out))
    layers = []
    layers.append(ResidualBlockDecoder(channels_in, channels_out, kernelsize, stride, padding, dilation, upsample))
    return nn.Sequential(*layers)




class TextClassifier(nn.Module):
    def __init__(self, num_features = 71, dim = 64 ):
        super(TextClassifier, self).__init__()
        self.conv1 = nn.Conv1d(num_features, 2 * dim, kernel_size=1);
        self.resblock_1 = make_res_block_encoder(2 * dim, 3 * dim, kernelsize=4, stride=2, padding=1,
                                                 dilation=1);
        self.resblock_4 = make_res_block_encoder(3 * dim, 2 * dim, kernelsize=4, stride=2, padding=0,
                                                 dilation=1);
        self.d = nn.d(p=0.5, inplace=False);
        self.linear = nn.Linear(in_features=2*dim, out_features=10, bias=True) # 10 is the number of classes
        self.sigmoid = nn.Sigmoid();


    def forward(self, x):
        x = x.transpose(-2,-1)
        h = self.conv1(x);
        h = self.resblock_1(h);
        h = self.resblock_4(h);
        h = self.d(h);
        h = h.view(h.size(0), -1);
        h = self.linear(h);
        out = self.sigmoid(h);
        return out;


    def get_activations(self, x):
        h = self.conv1(x);
        h = self.resblock_1(h);
        h = self.resblock_2(h);
        h = self.resblock_3(h);
        h = self.resblock_4(h);
        h = h.view(h.size(0), -1);
        return h;





class SVHN_Classifier_shie(nn.Module):
    def __init__(self):
        super(SVHN_Classifier_shie, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class MNIST_Classifier_shie(nn.Module):
    def __init__(self):
        super(MNIST_Classifier_shie, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def get_activation(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        #x = F.relu(self.fc1(x))
        #x = F.d(x, training=self.training)
        #x = self.fc2(x)
        return x
        
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)