

from src.unimodal.mhd.LabelVae import LabelDecoder, LabelEncoder
from torchvision.utils import save_image
import json
import torch
import torchvision
from src.abstract.modality import Modality
from src.utils import *
import torch.nn as nn
from src.unimodal.celeba.aes import encoder_att,decoder_att,Image_enc ,Image_dec
from sklearn.metrics import f1_score






class Image_mod(Modality):
    def __init__(self, latent_dim, size=[3,128,128], name="image", lhood_name="normal", enc=None, dec=None, 
                 reconstruction_weight=1 ,deterministic = True,distengled=False,latent_dim_w=None):

        super().__init__(latent_dim, size, name, enc,
                         dec, lhood_name, reconstruction_weight)

        enc_channel_list = [(64,128,128,2), (128,256,256,2), (256,512,512,2)]
        dec_channel_list = [(512,512,256,2), (256,256,128,2), (128,128,64,2)]

        self.enc = Image_enc(latent_dim=latent_dim,channel_list=enc_channel_list)
        self.dec = Image_dec(latent_dim=latent_dim,channel_list=dec_channel_list,enc_channel_list = enc_channel_list)

        self.classifier = None
        self.modality_type = "img"
        self.gen_quality = True
        self.file_suffix = ".png"
        
        self.fd = {
            "fd":"inception",
            "act_dim" : 2048
        }

    def save_output(self, output, filename):
        save_image(output.view(output.size(0), 3, 128, 128), filename+".png")
        
    
    def plot(self,x):
        return torchvision.utils.make_grid( x.view(x.size(0), 3, 128, 128), 8 ,pad_value=150,padding = 2)
        
    def reshape(self, x):
        return x.view(x.size(0),3,128,128)
    

    def save_data(self, d, fn, args):
        img_per_row = args['img_per_row']
        save_image(d.view(d.size(0), 3, 128, 128), fn, nrow=img_per_row)



class Mask_mod(Modality):
    def __init__(self, latent_dim, size=[1,128,128], name="mask", lhood_name="normal", enc=None, dec=None, 
                 reconstruction_weight=1 ,deterministic = False,distengled=False,latent_dim_w=None):

        super().__init__(latent_dim, size, name, enc,
                         dec, lhood_name, reconstruction_weight)
   
        enc_channel_list = [(64,128,128,4), (128,256,256,4)]
        dec_channel_list = [(256,256,128,4), (128,128,64,4)]

        self.enc = Image_enc(latent_dim=latent_dim,channel_list=enc_channel_list,img_ch=1,size_in=128)
        self.dec = Image_dec(latent_dim=latent_dim,channel_list=dec_channel_list,img_ch=1,size_in=128,enc_channel_list = enc_channel_list)


        self.classifier = None
        self.modality_type = "img"
        self.gen_quality = False
        self.file_suffix = ".png"
        self.fd = None

    def save_output(self, output, filename):
        save_image(output.view(output.size(0), 1, 128, 128), filename+".png")
        
    
    def plot(self,x):
        return torchvision.utils.make_grid( x.view(x.size(0), 1, 128, 128), 8 ,pad_value=150,padding = 2)
        
    def reshape(self, x):
        return x.view(x.size(0),1,128,128)
    

    def get_f_1_score(self,out,input):
        #sigmoid_outputs = torch.sigmoid(mask_outputs).cpu()
        #predicted_mask_round = np.round(out.cpu()) 

        #input_mask_round = np.round(input.cpu())

        #true_mask.append(input_mask_round.view(masks.shape[0],-1))
        #predicted_mask.append(predicted_mask_round.view(masks.shape[0],-1))
        #total_mask += torch.prod(torch.tensor(masks.shape))
        #correct_mask += (predicted_mask_round == input_mask_round.cpu()).sum().item()
        target = np.round( input.view(out.size(0),-1).cpu().numpy() ).astype(int)
        pred =np.round( out.view(out.size(0),-1).cpu().numpy() ).astype(int)

        # print("mask_f1")
        # print(target[0])
        # print(pred[0])
        # print(sum(target[0]))
        f1_avg = f1_score(target,pred, average='samples')
        # print(f1_avg)
        return  f1_avg
    

    def save_data(self, d, fn, args):
        img_per_row = args['img_per_row']
        save_image(d.view(d.size(0), 1, 128, 128), fn, nrow=img_per_row)


class Attributes(Modality):
    def __init__(self, latent_dim, size=[200], name="attributes", lhood_name="normal", 
                 enc=None, dec=None, reconstruction_weight=1 ,deterministic = False,distengled=False,latent_dim_w=None):
        super().__init__(latent_dim, size, name, enc,
                         dec, lhood_name, reconstruction_weight)

        self.enc = encoder_att(latent_dim=latent_dim)
        self.dec = decoder_att(latent_dim=latent_dim)
   
        #self.classifier = Trajectory_Classifier()
        
       # self.classifier.load_state_dict(torch.load(PATHS_CLASSIFIERS[self.name])['state_dict'])
       # self.classifier.eval()
        self.modality_type = "txt"
        self.gen_quality = False
        self.file_suffix = ".txt"
        
        #self.file_suffix = ".png"
        #self.fd = {
        #    "fd":"classifier",
        #    "act_dim" : 128
        #}
        
        
    #def save_output(self, output, filename):
    #    get_trajectories_image(output,filename = filename+self.file_suffix )
 
    def calc_log_prob(self, data, output, reduction="sum"):

        bce_logit_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        recon_loss = bce_logit_loss(output, data.float())
        return - recon_loss
    def plot(self,output):
        return str(output)
    
    def get_f_1_score(self,out,input):
        sigmoid_outputs = torch.sigmoid(out).cpu().numpy()
        predicted = np.round(sigmoid_outputs).astype(int) 
        #total = input.shape[0] * input.shape[1]
        #correct = (predicted == input.cpu()).sum().item()
        target = np.round( input.cpu().numpy()).astype(int)

        #print("attr_f1")
        #print(target[0])
        #print(predicted[0])
        #print(sum(predicted[0]))
        #print(sum(target[0]))
        f1_avg = f1_score(target, predicted, average='samples')
        #print(f1_avg)
        return  f1_avg
    
    def save_output(self, output, filename):
        return
    
    def reshape(self, x):
        return super().reshape(x)