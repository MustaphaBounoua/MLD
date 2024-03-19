

from src.unimodal.mhd.LabelVae import LabelDecoder, LabelEncoder
from torchvision.utils import save_image
import json
import torch
import torchvision

from src.unimodal.mhd.ImageVae import ImageDecoder,ImageEncoder,MHDImageEncoderPLus,ImageDecoderPlus
from src.unimodal.mhd.TrajectoryVae import TrajectoryDecoder,TrajectoryEncoder, MHDTrajEncoderPLus
from src.unimodal.mhd.SoundVae import SoundDecoder,SoundEncoder ,SigmaSoundEncoder, SigmaSoundDecoder,SoundDecoderPlus,MHDSoundEncoderPLus

from src.eval_metrics.Classifiers.MhdClassifiers import Sound_Classifier,Image_Classifier,Trajectory_Classifier

from src.unimodal.mnist_svhn.MnistVAE import MnistEncoder, MnistDecoder

from src.dataLoaders.MnistSvhnText.MnistSvhnText import tensor_to_text
from src.abstract.modality import Modality
from src.utils import *
from src.eval_metrics.Classifiers.MnistSvhnClassifiers import MnistClassifier,SVHNClassifier,TextClassifier
from src.unimodal.mnist_svhn.MnistVAE import MnistDecoder ,MnistEncoder
DATA_FOLDER = "./data/data_mnistsvhntext/"

alphabet_file = DATA_FOLDER+"alphabet.json"
LAPLACE_SCALE = 0.75


PATHS_CLASSIFIERS={
    "image":"data/data_mhd/clf/image_clf.pth.tar",
    "sound":"data/data_mhd/clf/sound_clf.pth.tar",
    "trajectory":"data/data_mhd/clf/trajectory_clf.pth.tar"       
}

traj_normalisation = {'max': 2.4120986461639404, 'min': -2.7722983360290527} 
sound_normalisation = {'max': -47.563736, 'min': -200.0}

class Image_mod(Modality):
    def __init__(self, latent_dim, size=[1,28,28], name="image", lhood_name="normal", enc=None, dec=None, 
                 reconstruction_weight=1 ,deterministic = False,distengled=False,latent_dim_w=None):
        super().__init__(latent_dim, size, name, enc,
                         dec, lhood_name, reconstruction_weight)
        if distengled == False:
            self.enc = ImageEncoder(latent_dim=latent_dim, deterministic= deterministic)
            self.dec = ImageDecoder(latent_dim=latent_dim)
        else:
            self.enc = MHDImageEncoderPLus(latent_dim=latent_dim, latent_dim_w=latent_dim_w)
            self.dec = ImageDecoderPlus(latent_dim=latent_dim+latent_dim_w)

     #   self.enc = MnistEncoder(latent_dim=latent_dim, deterministic= deterministic)
     #   self.dec = MnistDecoder(latent_dim=latent_dim)

        self.classifier = Image_Classifier()
       
        self.classifier.load_state_dict(torch.load(PATHS_CLASSIFIERS[self.name])['state_dict'])
        self.classifier.eval()
        self.modality_type = "img"
        self.gen_quality = "fid"
        self.file_suffix = ".png"
        self.fd = {
            "fd":"classifier",
            "act_dim" : 128
        }

    def save_output(self, output, filename):
        save_image(output.view(output.size(0), 1, 28, 28), filename+".png")
        
    
    def plot(self,x):
        return torchvision.utils.make_grid( x.view(x.size(0), 1, 28, 28), 8 ,pad_value=150,padding = 2)
        
    def reshape(self, x):
        return x.view(x.size(0),1,28,28)
    

    def save_data(self, d, fn, args):
        img_per_row = args['img_per_row']
        save_image(d.view(d.size(0), 1, 28, 28), fn, nrow=img_per_row)



class Trajectory(Modality):
    def __init__(self, latent_dim, size=[200], name="trajectory", lhood_name="normal", 
                 enc=None, dec=None, reconstruction_weight=1 ,deterministic = False,distengled=False,latent_dim_w=None):
        super().__init__(latent_dim, size, name, enc,
                         dec, lhood_name, reconstruction_weight)
        if distengled==False:
            self.enc = TrajectoryEncoder(latent_dim=latent_dim,deterministic= deterministic)
            self.dec = TrajectoryDecoder(latent_dim=latent_dim)
        else:
            self.enc = MHDTrajEncoderPLus(latent_dim=latent_dim, latent_dim_w=latent_dim_w)
            self.dec = TrajectoryDecoder(latent_dim=latent_dim+latent_dim_w)

        
        
        self.classifier = Trajectory_Classifier()
        
        self.classifier.load_state_dict(torch.load(PATHS_CLASSIFIERS[self.name])['state_dict'])
        self.classifier.eval()
        self.modality_type = "img"
        self.gen_quality = "fid"
        self.file_suffix = ".png"
        self.fd = {
            "fd":"classifier",
            "act_dim" : 128
        }
        
        
    def save_output(self, output, filename):
        get_trajectories_image(output,filename = filename+self.file_suffix , traj_norm =traj_normalisation)
 
       
    def plot(self,output):
        return get_trajectories_image(output,filename = "",traj_norm =traj_normalisation)
    
    
    def reshape(self, x):
        return x.view(x.size(0),3,32,32)
    
         
    
    def save_data(self, d, fn, args):
        save_one_trajectory(d.cpu(),fn,traj_norm=traj_normalisation)
    
    
class Sound(Modality):
    def __init__(self, latent_dim, size=[1,32,128], name="sound", lhood_name="normal", enc=None, dec=None, 
                 reconstruction_weight=1 ,pretrained = False,deterministic = False,distengled=False,latent_dim_w=None):
        super().__init__(latent_dim, size, name, enc,
                         dec, lhood_name, reconstruction_weight)
        if distengled==False:
            if pretrained :
                self.enc = SigmaSoundEncoder(latent_dim=latent_dim)
                self.dec = SigmaSoundDecoder(latent_dim=latent_dim)
            else:
                self.enc = SoundEncoder(latent_dim=latent_dim,deterministic= deterministic)
                self.dec = SoundDecoder(latent_dim=latent_dim)
        else:
            self.enc = MHDSoundEncoderPLus(latent_dim=latent_dim, latent_dim_w=latent_dim_w)
            self.dec = SoundDecoderPlus(latent_dim=latent_dim+latent_dim_w)




        self.classifier = Sound_Classifier()
        self.classifier.load_state_dict(torch.load(PATHS_CLASSIFIERS[self.name])['state_dict'])
        self.classifier.eval()
        self.modality_type = "audio"
        self.gen_quality = "fad"
        self.fad = True
        self.gen_quality = None
        self.file_suffix = ".wav"
        
    def save_output(self, output, filename):
        save_wave_sound(output,filename,sound_norm=sound_normalisation)
        
         
    def plot(self, output):
        output = output.view(-1,*self.size)
        return save_sound(output,filename="",sound_norm=sound_normalisation)

    def plot_spec(self,spec,legend = True):
        return plot_spectrogram(spec[0,:,:].permute(1,0), title=None, ylabel='freq_bin', aspect='auto', xmax=None)

    def save_data(self, d, fn, args=None):
        #img_per_row = args['img_per_row']
        save_wave_sound(d , filename = fn, sound_norm = sound_normalisation)


    def reshape(self, x):
        return x


class LABELmhd(Modality):
    def __init__(self, latent_dim, size=8*71, name="label", lhood_name="categorical", enc=None, dec=None, 
                 reconstruction_weight=1,deterministic = False):
        super().__init__(latent_dim, size, name, enc,
                         dec, lhood_name, reconstruction_weight)

        self.enc = LabelEncoder(latent_dim=latent_dim,deterministic= deterministic)
        self.dec = LabelDecoder(latent_dim=latent_dim)
        
        
        self.classifier = TextClassifier()
        self.classifier.load_state_dict(torch.load(PATHS_CLASSIFIERS[self.name]))
        self.classifier.eval()
        self.modality_type = "txt"
        self.gen_quality = None
        
    def save_output(self, output, filename):
        text = tensor_to_text(self.alphabet, output)
        write_samples_text_to_file(text, filename+".txt")
        
         
    def plot(self, output):
        text = tensor_to_text(self.alphabet, output)
        return str(text)
        


    def reshape(self, x):
        return x