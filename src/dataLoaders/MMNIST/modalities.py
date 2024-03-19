from torchvision.utils import save_image
import json
import torch
import torchvision

from src.unimodal.mmnist.mmnist_vae_plus import Enc, Dec
from src.abstract.modality import Modality
from src.eval_metrics.Classifiers.MMnistClassifiers import ClfImg
from src.unimodal.mmnist.mmnist_vae import EncoderImg, DecoderImg, MMNISTEncoderplus,DecoderImgplus


PATHS_CLASSIFIERS = "data/MMNIST/clf/pretrained_img_to_digit_clf_"

class MMNIST(Modality):
    def __init__(self, latent_dim, size=[3,28,28], name="m0", 
                 lhood_name="laplace", enc=None, dec=None, reconstruction_weight=1 , 
                 deterministic = False, 
                 distengled= False,
                 latent_dim_w= None,resnet= False):
        
        super().__init__(latent_dim, size, name, enc,
                         dec, lhood_name, reconstruction_weight)
       # self.enc = MnistEncoder(latent_dim=latent_dim, deterministic = deterministic)
       # self.dec = MnistDecoder(latent_dim=latent_dim)
        if distengled == False:
            if resnet==False:
                self.enc = EncoderImg(latent_dim=latent_dim, deterministic= deterministic)
                self.dec = DecoderImg(latent_dim=latent_dim)
            else:
                self.enc = Enc(latent_dim=latent_dim,deterministic= deterministic)
                self.dec = Dec(latent_dim=latent_dim)
        else:
            if resnet:
                self.enc = Enc(latent_dim=latent_dim, ndim_w=latent_dim_w, distengled=True,deterministic= False)
                self.dec = Dec(latent_dim=latent_dim+latent_dim_w)
            else:
                print("here")
                self.enc = MMNISTEncoderplus(latent_dim=latent_dim, latent_dim_w=latent_dim_w)
                self.dec = DecoderImgplus(latent_dim=latent_dim+latent_dim_w)


        self.classifier = ClfImg()
        self.classifier.load_state_dict(torch.load(PATHS_CLASSIFIERS+str(self.name)))
        self.classifier.eval()
        self.modality_type = "img"
        self.gen_quality = "fid"
        self.file_suffix = ".png"
        self.fd = {
            "fd":"inception",
            "act_dim" : 2048
        }



    def save_output(self, output, filename):
        save_image(output.view(output.size(0), 3, 28, 28), filename+".png")
        
    def plot(self,x):
        return torchvision.utils.make_grid( x.view(x.size(0), 3, 28, 28), 8  )
        
    def reshape(self, x):
        return x.view(x.size(0),3,28,28)
    
     
    def save_data(self, d, fn, args):
        img_per_row = args['img_per_row']
        save_image(d.data.cpu(), fn, nrow=img_per_row);






    
    
    
