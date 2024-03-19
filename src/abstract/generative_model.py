from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class GenerativeModel(ABC, nn.Module):
    def __init__(self):
         super(GenerativeModel, self).__init__()
         
         
    @abstractmethod
    def conditional_gen_all_subsets(self, x): raise NotImplementedError
    
    
    @abstractmethod
    def gen_latent(self,x): raise NotImplementedError
    
    
    @abstractmethod
    def conditional_gen_latent_subsets(self, x): raise NotImplementedError