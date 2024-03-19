from torchvision.utils import save_image
import json
import torch
import torchvision
import numpy as np
import PIL
from src.unimodal.mnist_svhn.MnistVAE import MnistDecoder, MnistEncoder ,MnistEncoderplus
from src.unimodal.mnist_svhn.MnistLabelVAE import TextDecoder, TextEncoder
from src.unimodal.mnist_svhn.SvhnVae import SvhnDecoder,SvhnEncoder,SvhnEncoder_plus
from src.dataLoaders.MnistSvhnText.MnistSvhnText import tensor_to_text
from src.abstract.modality import Modality
from src.utils import write_samples_text_to_file
from src.eval_metrics.Classifiers.MnistSvhnClassifiers import TextClassifier ,SVHN_Classifier_shie,MNIST_Classifier_shie
from src.unimodal.mhd.ImageVae import ImageDecoder,ImageEncoder
from torchvision import transforms

#from pytorchcv.model_provider import get_model as ptcv_get_model

DATA_FOLDER = "./data/data_mnistsvhntext/"

alphabet_file = DATA_FOLDER+"alphabet.json"
LAPLACE_SCALE = 0.75



PATHS_CLASSIFIERS={
    "mnist":"data/data_mnistsvhntext/clf/MNIST_classifier.pt",
    "svhn":"./data/data_mnistsvhntext/clf/svhn_classifier.pt",
    "label":"./data/data_mnistsvhntext/clf/clf_text"       
}


class MNIST(Modality):
    def __init__(self, latent_dim, size=[1,28,28], name="mnist", 
                 lhood_name="laplace", enc=None, dec=None,
                 reconstruction_weight=1 , deterministic = False,latent_code=None,
                 convnet= False,distengled=False,latent_dim_w=None):
        super().__init__(latent_dim, size, name, enc,
                         dec, lhood_name, reconstruction_weight)

        # self.enc = Encoder(latent_dim=latent_dim, deterministic= deterministic)
        # self.dec = Decoder(latent_dim=latent_dim)             
        if distengled ==True:
            self.enc = MnistEncoderplus(latent_dim=latent_dim, latent_dim_w=latent_dim_w)
            self.dec = MnistDecoder(latent_dim=latent_dim + latent_dim_w)

        elif latent_code == None:
            if convnet:
                self.enc = ImageEncoder(latent_dim=latent_dim, deterministic= deterministic)
                self.dec = ImageDecoder(latent_dim=latent_dim)
            else:
                self.enc = MnistEncoder(latent_dim=latent_dim, deterministic = deterministic)
                self.dec = MnistDecoder(latent_dim=latent_dim)
        else:
            if convnet:
                self.enc = ImageEncoder(latent_dim=latent_dim, deterministic= deterministic)
                self.dec = ImageDecoder(latent_dim=latent_code)
            else:
                self.enc = MnistEncoder(latent_dim=latent_dim, deterministic = deterministic)
                self.dec = MnistDecoder(latent_dim=latent_code)

       # self.classifier = MnistClassifier()
        self.classifier = MNIST_Classifier_shie()
        self.classifier.load_state_dict(torch.load(PATHS_CLASSIFIERS[self.name]))
        self.classifier.eval()
        self.modality_type = "img"
        self.gen_quality = "fid"
        self.file_suffix = ".png"
        self.fd = {
            "fd":"classifier",
            "act_dim" : 320
        }


    def save_output(self, output, filename):
        save_image(output.view(output.size(0), 1, 28, 28), filename+".png")
        
    
    def plot(self,x):
        return torchvision.utils.make_grid( x.view(x.size(0), 1, 28, 28),8  )
        
    def reshape(self, x):
        return x.view(x.size(0),1,28,28)
    
     
    def save_data(self, d, fn, args):
        img_per_row = args['img_per_row']
        save_image(d.data.cpu(), fn, nrow=img_per_row);
    # def save_data(self, d, fn, args):
    #     # img_per_row = args['img_per_row']
    #     # save_image(d.data.cpu(), fn, nrow=img_per_row)
        
        
    #     tensor = d.data.cpu().permute(1,2,0).numpy() *255.0
    #     # print('okay')
    #     # print( (tensor == (d.data.cpu().permute(1,2,0) *255.0 ).numpy() ).sum()  )
    #     tensor = np.array(tensor, dtype=np.uint8)
    #     cv2.imwrite(fn, tensor, [cv2.IMWRITE_PNG_COMPRESSION, 0])



class SVHN(Modality):
    def __init__(self, latent_dim, size=[3,32,32], name="svhn", lhood_name="laplace", 
                 enc=None, dec=None, reconstruction_weight=1, deterministic = False,latent_code=None,distengled=False,latent_dim_w=None):
        super().__init__(latent_dim, size, name, enc,
                         dec, lhood_name, reconstruction_weight)
        

        if distengled ==True:
            self.enc = SvhnEncoder_plus(latent_dim=latent_dim, latent_dim_w=latent_dim_w)
            self.dec = SvhnDecoder(latent_dim=latent_dim + latent_dim_w)
      
        else:
                self.enc = SvhnEncoder(latent_dim=latent_dim,deterministic = deterministic)
                self.dec = SvhnDecoder(latent_dim=latent_dim)
        


        self.classifier = SVHN_Classifier_shie()
       # self.classifier = self.get_resnet_svhn()
        self.classifier.load_state_dict(torch.load(PATHS_CLASSIFIERS[self.name]))
        self.classifier.eval()
        self.modality_type = "img"
        self.gen_quality = "fid"
        self.file_suffix = ".png"
        self.img_size = torch.Size((3, 28, 28))
        self.transform_plot = self.get_transform()
        self.fd = {
            "fd":"inception",
            "act_dim" : 2048
        }
        
    # def get_resnet_svhn(self):
    #     net = ptcv_get_model("resnet20_svhn", pretrained=True)
    #     return net   
                            
    def save_output(self, output, filename):
       # transformed = self.transform_plot( output.view(output.size(0), 3, 32, 32))
        save_image(output.view(output.view(output.size(0), 3, 32, 32), filename+".png"))
        
       
    
    def plot(self,x):
        # transformed = torch.Tensor()
        # for img in range(0,x.size(0) ):
        #     transformed = torch.cat([transformed,self.transform_plot(x[img,:,:,:]).view(1,3,32,32)])
   
        return torchvision.utils.make_grid( torch.clamp(x,min =0.0, max =1.0) , 8  )
    
    
    def reshape(self, x):
        return x.view(x.size(0),3,32,32)
    
    def tensor_to_image(self,d):
        d = torch.clamp(d,min =0.0, max =1.0)
        tensor = d.data.cpu().permute(1,2,0) *255
        tensor = np.array(tensor, dtype=np.uint8)
       # print(tensor.shape)
        return PIL.Image.fromarray(tensor)

    def save_data(self, d, fn, args):
        #img_per_row = args['img_per_row']
        save_image(d.data.cpu(), fn, nrow=1)
        
        
        # tensor = d.data.cpu().permute(1,2,0).numpy() *255.0
        # # print('okay')
        # # print( (tensor == (d.data.cpu().permute(1,2,0) *255.0 ).numpy() ).sum()  )
        # tensor = np.array(tensor, dtype=np.uint8)
        # cv2.imwrite(fn, tensor, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    #    # image = self.tensor_to_image(d)
    #    # image.save(fn)
    #     print(fn)
    #     print( (cv2.imread(fn) == tensor ).sum()  )

        
    def get_transform(self):
        transf = transforms.Compose([transforms.ToPILImage(),
                                    #  transforms.Resize(size=list(self.img_size)[1:],
                                    #                    interpolation=Image.BICUBIC),
                                     transforms.ToTensor()])
        return transf

    
    
class LABEL(Modality):
    def __init__(self, latent_dim, size=8*71, name="label", lhood_name="categorical",
                 enc=None, dec=None, reconstruction_weight=1, deterministic = False):
        super().__init__(latent_dim, size, name, enc,
                         dec, lhood_name, reconstruction_weight)

        self.enc = TextEncoder(latent_dim=latent_dim,deterministic = deterministic)
        self.dec = TextDecoder(latent_dim=latent_dim)
        
        with open(alphabet_file) as al_file:
            alphabet = str(''.join(json.load(al_file)))
        self.alphabet= alphabet
        self.classifier = TextClassifier()
        self.classifier.load_state_dict(torch.load(PATHS_CLASSIFIERS[self.name])).eval()
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





