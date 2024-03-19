import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset, DataLoader
import torchvision
import glob
import os
from PIL import Image

attr_visible  = [4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35]

class CelebAHQMaskDS(Dataset):
    def __init__(self, size=128, datapath='/home/bounoua/work/mld/data/data_celba/CelebAMask-HQ/',  train = True, all_mod ="all"):
        """
            Args: 
                datapath: folder path containing train, val, and test folders of images and mask and celeba attribute text file
                transform: torchvision transform for the images and masks
                ds_type: train, val, or test
        """

        super().__init__()
        self.size = size
        self.all_mod =all_mod
        self.datapath = datapath
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize([size, size]),
                            torchvision.transforms.ToTensor(),
                            ])

      
        d_image ="/home/bounoua/work/mld/data/data_celba/CelebAMask-HQ/CelebA-HQ-img"
        d_mask = "/home/bounoua/work/mld/data/data_celba/CelebAMask-HQ/CelebAMaskHQ-mask"
  

        self.img_files = [d_image + "/"+ p for p in os.listdir(d_image) ]
        self.mask_files = [d_mask + "/"+ p for p in os.listdir(d_mask) ]

        self.img_files.sort()
        self.mask_files.sort()
        assert len(self.img_files) == len(self.mask_files)
        
        self.attr_tensor = torch.zeros((len(self.img_files),40), dtype=int)
        self.img_tensor = torch.zeros(len(self.img_files),3,self.size,self.size)
        self.mask_tensor = torch.zeros(len(self.img_files),1,self.size,self.size)
        
        # Read attr text file
        attr_txt_file = open(self.datapath + 'CelebAMask-HQ-attribute-anno.txt')
        attr_list = attr_txt_file.readlines()
        self.attributes = attr_list[1].strip().split(" ")
        assert len(self.attributes) == 40
        
        # for i in range(len(self.img_files)):
        #     assert self.img_files[i].split("/")[-1][:-4] == self.mask_files[i].split("/")[-1][:-4]

        #     img_idx = int(self.img_files[i].split("/")[-1][:-4])
        #     attr_i = attr_list[img_idx + 2].strip().split(" ")
        #     assert img_idx == int(attr_i[0][:-4])
        #     attr_i01 = torch.tensor([1 if a == '1' else 0 for a in attr_i[2:]])
        #     self.attr_tensor[i] = attr_i01

        # torch.save(self.attr_tensor, datapath +"att.pth")
        self.attr_tensor = torch.load("/home/bounoua/work/mld/data/data_celba/CelebAMask-HQ/att.pth")[:, attr_visible]
        N = 25000
        if train :
            self.img_files = self.img_files [:N]
            self.mask_files = self.mask_files[:N]
            self.attr_tensor = self.attr_tensor[:N]
        else: 
            self.img_files = self.img_files [N:]
            self.mask_files = self.mask_files[N:]
            self.attr_tensor = self.attr_tensor[N:]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        """
        Returns a tuple of image, mask, attribute
        """
        if self.all_mod =="all":
            im = self.transform(Image.open(self.img_files[index]))
            mask = self.transform(Image.open(self.mask_files[index]))
            return { "image" : im, "mask" :mask,"attributes": self.attr_tensor[index]}, True
        else:
            if self.all_mod =="image":
                return { "image" : self.transform(Image.open(self.img_files[index]))}, True
            elif self.all_mod =="attributes":
                return { "attributes" : self.attr_tensor[index]}, True
            elif self.all_mod =="mask":
                return { "mask" : self.transform(Image.open(self.mask_files[index] ))}, True
            



def read_tensor(file_path):
    return torch.load(file_path)


class Dataset_latent(Dataset):
    def __init__(self, folder = "/home/bounoua/work/mld/data/data_celba/latent/",train =True,im = 128):
            if train:
                self.imgs = read_tensor(folder+"/train/"+"image_{}.pth".format(im))  
                self.mask = read_tensor(folder+"/train/"+"mask_64.pth") 
                self.att = read_tensor(folder+"/train/"+"att_16.pth") 
            else:
                self.imgs = read_tensor(folder+"/test/"+"image_{}.pth".format(im))  
                self.mask = read_tensor(folder+"/test/"+"mask_64.pth") 
                self.att = read_tensor(folder+"/test/"+"att_16.pth") 

    def __getitem__(self, i):
        
        return { "image" : self.imgs [i], "mask" :self.mask [i],"attributes": self.att[i]}

    def __len__(self):
        return self.imgs.size(0) 
    





class Dataset_latent_2(Dataset):
    def __init__(self, folder = "/home/bounoua/work/mld/data/data_celba/latent_2/",train =True,im = 256):
            if train:
                self.imgs = read_tensor(folder+"/train/"+"image_{}.pth".format(im))  
                self.mask = read_tensor(folder+"/train/"+"mask_128.pth") 
                self.att = read_tensor(folder+"/train/"+"att_32.pth") 
            else:
                self.imgs = read_tensor(folder+"/test/"+"image_{}.pth".format(im))  
                self.mask = read_tensor(folder+"/test/"+"mask_128.pth") 
                self.att = read_tensor(folder+"/test/"+"att_32.pth") 

    def __getitem__(self, i):
        
        return { "image" : self.imgs [i], "mask" :self.mask [i],"attributes": self.att[i]}

    def __len__(self):
        return self.imgs.size(0)