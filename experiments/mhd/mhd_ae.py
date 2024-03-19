

import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import logging

from src.dataLoaders.MHD.MHD import MHDDataset
from src.dataLoaders.MHD.modalities import Image_mod,Sound,Trajectory

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
logging.getLogger("lightning").setLevel(logging.ERROR)
from src.utils import create_forlder
import os

from src.unimodal.BaseAE import AE

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)


parser.add_argument('--modality', type=str, default="image",
                    help=("Options possible are image, sound, trajectory"))



NUM_epoch = 1
batch_size = 64
beta = 1.0

lr =1e-4


train,test = MHDDataset(train=True),MHDDataset(train=False)


train_loader = DataLoader(train, batch_size=batch_size,
                          shuffle=True,
                          num_workers=4, drop_last=True)

test_loader = DataLoader(test, batch_size=batch_size,
                          shuffle= True,
                          num_workers=2, drop_last=True)




CHECKPOINT_DIR = "trained_models/"
create_forlder(CHECKPOINT_DIR)
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'MHD')
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'AE')

if __name__ == "__main__":
    
    args = parser.parse_args()
    dim_sound = 128
    dim_image = 64
    dim_traj = 16 
    
    if args.modality =="image":
        dim_latent = dim_image
        modality=   Image_mod(latent_dim=dim_image, deterministic = True, lhood_name="normal" )
                
    elif args.modality =="sound":
        dim_latent = dim_sound
        modality =Sound(latent_dim=dim_sound, pretrained = False, deterministic =True , lhood_name="normal" )
                            
    elif args.modality =="trajectory":
        dim_latent = dim_traj
        modality = Trajectory(latent_dim= dim_traj,deterministic = True,  lhood_name="normal" ) 

    model = AE(modality= modality,
                     train_samp = next(iter(train_loader))[0][args.modality],
                        test_samp=next(iter(test_loader))[0][args.modality],
               regularization= None, lr = lr)
    

    print(model)
    
    tb_logger =  TensorBoardLogger(save_dir = CHECKPOINT_DIR,
                                    name=str("ae_deter"+args.modality )
                                    )

    trainer = pl.Trainer(
        logger= tb_logger, 
        check_val_every_n_epoch=1,
        accelerator='gpu', 
        devices= 1  ,
        max_epochs= NUM_epoch, 
        default_root_dir = CHECKPOINT_DIR,
)


        
        
        
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders= test_loader )

    