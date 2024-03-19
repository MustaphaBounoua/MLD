import torch
import torch.distributions as dist
from abc import ABC, abstractmethod
import torch.nn.functional as F

DATA_FOLDER = "./data/data_mnistsvhntext/"

alphabet_file = DATA_FOLDER+"alphabet.json"
LAPLACE_SCALE = 0.75


class Modality(ABC):
    def __init__(self, latent_dim, size, name, enc, dec, lhood_name, reconstruction_weight,laplace_scale =LAPLACE_SCALE):
        self.name = name
        self.size = size
        self.enc = enc
        self.dec = dec
        self.laplace_scale = laplace_scale
        self.latent_dim = latent_dim
        self.likelihood_name = lhood_name
        self.likelihood = self.get_likelihood(lhood_name)
        self.reconstruction_weight = reconstruction_weight
        self.classifier = None
        self.modality_type = 'None'
        self.gen_quality = None
        self.fad = False
        

    def get_likelihood(self, name):
        
        if name == 'laplace':
            px = dist.Laplace
            self.scale = torch.tensor(self.laplace_scale)
        elif name == 'bernoulli':
            px = dist.Bernoulli
        elif name == 'normal':
            px = dist.Normal
        elif name == 'normal_logporb':
            px = dist.Normal
        elif name == 'categorical':
            px = dist.OneHotCategorical
        else:
            print('likelihood not implemented')
            px = None
        return px



    def calc_log_prob(self, data, output,reduction = "sum"):
        
        if self.likelihood_name == 'laplace':
            log_prob = self.likelihood(output, self.scale.type_as(data)).log_prob(data)
      
            if reduction == "sum":
                log_prob=log_prob.sum()
        elif self.likelihood_name == 'normal':
            log_prob = -  F.mse_loss(output.reshape(output.size(0), -1), data.reshape(output.size(0), -1), reduction='sum')
        elif self.likelihood_name == 'normal_logporb':
            log_prob = self.likelihood(output, self.scale.type_as(data)).log_prob(data).sum()
        else:
            #data = F.one_hot(torch.Tensor(data).long(),num_classes=1590)
            log_prob = self.likelihood(probs=output).log_prob(data)
           
            if reduction == "sum":
                log_prob = log_prob.sum()
          
        return log_prob
    
    
    @abstractmethod
    def save_output(self, output):
        pass
    
    @abstractmethod
    def plot(self, output):
        pass
    
    
    @abstractmethod
    def reshape(self,x):
        pass
    
    
    def classify(self,output, device = "cpu"):
        self.classifier.to(device)
        self.classifier.eval()
        #out = self.classifier(self.reshape(output) )
        out = self.classifier(output)
        labels = torch.argmax(out, axis=1)
        return labels
    

    def get_reconstruction(self, x):
        
        if self.likelihood_name == 'laplace':
            return self.likelihood(x, self.scale.type_as(x)).mean
        elif self.likelihood_name == 'categorical':
            return  self.likelihood(x).mean
        elif self.likelihood_name == 'normal':
            return x
 