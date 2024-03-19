
from torchvision.utils import save_image
import json
import torch
import torchvision
import textwrap
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from torchvision import transforms

from src.unimodal.CUB.text_resnet_limit import DecoderText, EncoderText
from src.unimodal.CUB.image_resnet_limit import DecoderImg, EncoderImg


from src.unimodal.CUB.cub_bird_plus import Enc as EncoderImagePlus
from src.unimodal.CUB.cub_bird_plus import Dec as DecoderImagePlus
from src.unimodal.CUB.cub_sentence_plus import Enc as EncoderSentencePlus
from src.unimodal.CUB.cub_sentence_plus import Dec as DecdoerSentencePlus


from src.unimodal.CUB.enhancer_ae import Encoder, Decoder
from src.abstract.modality import Modality
from src.utils import *

DATA_FOLDER = "./data/data_mnistsvhntext/"

LAPLACE_SCALE = 0.75

h =128

ddconfig = dict(
        double_z=True,
        z_channels=1,
        resolution=128,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2 , 4, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        embed_dim = 1
    )

ddconfig_64 = dict(
        double_z=True,
        z_channels=1,
        resolution=64,
        in_channels=3,
        out_ch=3,
        ch=64,
        ch_mult=[2, 2 , 4, 8],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        embed_dim = 1
    )
  


class Bird_Image(Modality):
    def __init__(self, latent_dim, h= 64, size=(3,64,64),  name="image", lhood_name="normal", enc=None, dec=None, 
                 reconstruction_weight=1 ,deterministic = False, resnet = False,laplace_scale = 0.75,
                   distengeled= False, latent_dim_w =None):
        super().__init__(latent_dim, size, name, enc,
                         dec, lhood_name, reconstruction_weight, laplace_scale = laplace_scale)
        config =ddconfig
        
        self.h = h
        if h == 64:
            config =ddconfig_64
        if resnet=="autokl":
            self.enc =   Encoder(**config)
            self.dec = Decoder(**config)
        elif resnet=="resnet":
            self.enc = EncoderImg(latent_dim=latent_dim, deterministic= deterministic)
            self.dec = DecoderImg(latent_dim=latent_dim)
        elif resnet=="resnetplus":
            if distengeled == False:
                self.enc = EncoderImagePlus(latent_dim=latent_dim, deterministic= deterministic)
                self.dec = DecoderImagePlus(latent_dim=latent_dim)
            else:
                self.enc = EncoderImagePlus(latent_dim=latent_dim, deterministic= False, distengeled= True, latent_dim_w = latent_dim_w)
                self.dec = DecoderImagePlus(latent_dim=latent_dim+latent_dim_w)
  
            
        self.classifier = None
        #self.classifier.load_state_dict(torch.load(PATHS_CLASSIFIERS[self.name])['state_dict'])
   
        self.modality_type = "img"
        self.gen_quality = "fid"
        self.file_suffix = ".png"
        self.fd = {
            "fd":"inception",
            "act_dim" : 2048
        }


    def save_output(self, output, filename):
        save_image(output.view(output.size(0), 3, self.h, self.h), filename+".png")
        
    
    def plot(self,x):
        
        return torchvision.utils.make_grid( x.view(x.size(0), 3, self.h,self.h), 8  )
        
    def reshape(self, x):
        return x.view(x.size(0),3,self.h,self.h)
    

    def save_data(self, d, fn, args):
        #img_per_row = args['img_per_row']
        save_image(d.view(1, 3, self.h,self.h), fn, nrow=1)









def load_vocab():
        # call dataloader function to create vocab file
        vocab_file ="data/data_cub/oc_3_msl_32/cub.vocab"
        with open(vocab_file, 'r') as vocab_file:
            vocab = json.load(vocab_file)
        return vocab['i2w']















class Sentence(Modality):
    
    def __init__(self, latent_dim, size=(32,1590), name="sentence", lhood_name="categorical", enc=None, dec=None, 
                 reconstruction_weight=1,deterministic = False, resnet=False,distengeled= False,latent_dim_w =None):
        
        super().__init__(latent_dim, size, name, enc,
                         dec, lhood_name, reconstruction_weight)
        
        if resnet=="resnet":
            self.enc = EncoderText(latent_dim=latent_dim,deterministic= deterministic)
            self.dec = DecoderText(latent_dim=latent_dim)
        else:
            if distengeled == False:
                self.enc = EncoderSentencePlus(latent_dim=latent_dim,latent_dim_w=latent_dim_w,deterministic= deterministic,distengeled=distengeled)
                self.dec = DecdoerSentencePlus(latent_dim=latent_dim,latent_dim_w=latent_dim_w,distengeled=distengeled)
            else:
                self.enc = EncoderSentencePlus(latent_dim=latent_dim,latent_dim_w=latent_dim_w,deterministic= deterministic,distengeled=distengeled)
                self.dec = DecdoerSentencePlus(latent_dim=latent_dim,latent_dim_w=latent_dim_w,distengeled=distengeled)

        
        self.font = ImageFont.truetype('FreeSerif.ttf', 18)
        self.classifier = None
        #self.classifier.load_state_dict(torch.load(PATHS_CLASSIFIERS[self.name])['state_dict'])
        self.vocab = load_vocab()
        self.modality_type = "txt"
        self.gen_quality = None
        self.file_suffix = ".txt"
        
        
    def save_output(self, output, filename):
        text = self.plot(output)
        write_samples_text_to_file(text, filename+".txt")
    
    def get_str(self,tensor):
        if tensor.size(-1) !=32:
            tensor = torch.argmax(tensor,dim=-1)
      
        
        fn_2i = lambda t: t.cpu().numpy().astype(int)
        fn_trun = lambda s: s[:np.where(s == 2)[0][0] + 1] if 2 in s else s
        
        data = fn_2i(tensor)
        data = [fn_trun(d) for d in data]
        i2w = self.vocab
        sentences_text =[]
        for d_sent in data:
            word_list = ' '.join(i2w[str(i)] for i in d_sent)
            word_list= word_list.replace('<eos>','').replace('<pad>','').replace('<unk>','').replace('<exc>','')
         
            sentences_text.append( word_list )
            #print('[DATA]  ==> {}'.format())    
        return sentences_text     
         
         
    def plot(self, output):
        
        if output.size(-1) !=32:
            output = torch.argmax(output,dim=-1)
      
        
        fn_2i = lambda t: t.cpu().numpy().astype(int)
        fn_trun = lambda s: s[:np.where(s == 2)[0][0] + 1] if 2 in s else s
        
        data = fn_2i(output)
        data = [fn_trun(d) for d in data]
        i2w = self.vocab
        sentences_text =[]
        for d_sent in data:
            sentences_text.append(' '.join(i2w[str(i)] for i in d_sent) )
            #print('[DATA]  ==> {}'.format())    
        return str(sentences_text)       
       # return self.grid_images(sentences_text)
        
    def grid_images(self,sentences):   
        grid = torch.Tensor()
        for sent in sentences:
             grid = torch.cat( [grid, self.text_to_pil(sent).view(1,3,300,300) ],)
        
        return torchvision.utils.make_grid(grid, 8  )
            
        
        
    def text_to_pil(self,text, w=300, h=300, linewidth=30):
        
        blank_img = torch.ones([3, w, h]);
        pil_img = transforms.ToPILImage()(blank_img.cpu()).convert("RGB")
        draw = ImageDraw.Draw(pil_img)
        text_sample = text
            
        lines = textwrap.wrap(''.join(text_sample), width=linewidth)
        y_text = h
        num_lines = len(lines);
        for l, line in enumerate(lines):
            width, height = self.font.getsize(line)
            draw.text((0, (h/2) - (num_lines/2 - l)*height), line, (0, 0, 0), font=self.font)
            y_text += height
       
        text_pil = transforms.ToTensor()(pil_img)
        return text_pil

 

    def reshape(self, x):
        return x
    
