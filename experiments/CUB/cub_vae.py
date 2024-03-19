import os
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import logging
from torch.utils.tensorboard import SummaryWriter
import sys
root = "/home/bounoua/work/mld/"
sys.path.append(root)

from src.dataLoaders.CUB.CUB import CubDataset ,CubDatasetImage,CubDatasetText
from src.dataLoaders.CUB.modalities import Sentence, Bird_Image

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
logging.getLogger("lightning").setLevel(logging.ERROR)
from src.utils import create_forlder
import os

from src.unimodal.vae import AE

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)


parser.add_argument('--modality', type=str, default="image",
                    help=("Options possible sentence"))



NUM_epoch = 1000
batch_size = 64
beta = 1.0

lr =1e-3


train,test = CubDataset(train=True),CubDataset(train=False)




CHECKPOINT_DIR = "trained_models/"
create_forlder(CHECKPOINT_DIR)
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'CUB')
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'vae')

if __name__ == "__main__":
    
    args = parser.parse_args()
    dim_image = 64
    dim_sentence = 32
    
    if args.modality =="image":
        decay = 0
        train,test = CubDatasetImage(train=True,data_augment= False,resize= True),CubDatasetImage(train=False,resize= True)
        train_loader = DataLoader(train, batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=8, drop_last=True ,pin_memory =True)
        test_loader = DataLoader(test, batch_size=batch_size,
                                  shuffle= False,
                                  num_workers=8, drop_last=True,pin_memory =True)
        dim_latent = dim_image
        modality=   Bird_Image(latent_dim=dim_image,
                                deterministic = False,distengeled=False, lhood_name="laplace",resnet="resnet",h=64 )
        
    elif args.modality =="sentence":
        decay = 0
        train,test = CubDatasetText(train=True),CubDatasetText(train=False)
        train_loader = DataLoader(train, batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4, drop_last=True ,pin_memory =True)
        test_loader = DataLoader(test, batch_size=batch_size,
                                  shuffle= False,
                                  num_workers=4, drop_last=True,pin_memory =True)
        dim_latent = dim_sentence
        modality = Sentence(latent_dim=dim_sentence, deterministic =True ,distengeled=False,  lhood_name="categorical",resnet= "transformer" )
           
    model = AE(modality= modality,
               test_loader= test_loader, decay = decay,train_loader = train_loader,
               regularization= None, lr = lr)
    

    print(model)
    
    tb_logger =  TensorBoardLogger(save_dir = CHECKPOINT_DIR,
                                    name=str("vae_{}_{}{}".format(dim_latent,args.modality,decay) )
                                    )

    trainer = pl.Trainer(
        logger= tb_logger, 
        check_val_every_n_epoch=10,
        accelerator='gpu', 
        devices= 1 ,
        max_epochs= NUM_epoch, 
        default_root_dir = CHECKPOINT_DIR,
          num_sanity_val_steps=0,
        #strategy="ddp",
      #  resume_from_checkpoint="/home/renault/Documents/code/mld/trained_models/CUB/AE_mmvaeplus/ae_model_aeautokhl64_256_image0/version_2/checkpoints/epoch=299-step=82800.ckpt"
)


    print(model)
        
        
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders= test_loader )

    