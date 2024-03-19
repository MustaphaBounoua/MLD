import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import logging
from src.dataLoaders.celebA.CelebA import CelebAHQMaskDS
from src.dataLoaders.celebA.modalities import Image_mod, Mask_mod, Attributes
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
logging.getLogger("lightning").setLevel(logging.ERROR)
from src.utils import create_forlder
import os

from src.unimodal.BaseAE import AE

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)


parser.add_argument('--modality', type=str, default="image",
                    help=("Options possible sentence"))



NUM_epoch = 1000
batch_size = 64


lr =1e-3





CHECKPOINT_DIR = "trained_models/"
create_forlder(CHECKPOINT_DIR)
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'CelebA')
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'AES_1')

if __name__ == "__main__":
    
    args = parser.parse_args()
    dim_image = 256
    dim_att= 32
    dim_mask = 128
    train,test = CelebAHQMaskDS(train=True),CelebAHQMaskDS(train=False)



    if args.modality =="image":
        NUM_epoch = 300
        batch_size = 64
        train,test = CelebAHQMaskDS(train=True, all_mod ="image" ),CelebAHQMaskDS(train=False,all_mod ="image")
        dim_latent = dim_image
        modality=   Image_mod(latent_dim=dim_image )
                
    elif args.modality =="mask":
        NUM_epoch = 200
        batch_size = 64
        train,test = CelebAHQMaskDS(train=True, all_mod ="mask" ),CelebAHQMaskDS(train=False,all_mod ="mask")
        dim_latent = dim_mask
        modality=   Mask_mod(latent_dim=dim_mask ) 

    elif args.modality =="att":
        NUM_epoch = 100
        batch_size = 64
        train,test = CelebAHQMaskDS(train=True, all_mod ="attributes" ),CelebAHQMaskDS(train=False,all_mod ="attributes")
        dim_latent = dim_att
        modality=   Attributes(latent_dim=dim_att  )  
    
    train_loader = DataLoader(train, batch_size=batch_size,
                          shuffle=True,pin_memory =True,
                          num_workers=4, drop_last=True)

    test_loader = DataLoader(test, batch_size=batch_size,
                          shuffle= False,pin_memory =True,
                          num_workers=4, drop_last=True)

    # print(len(train))
    # print(len(test))
    train_samp= next(iter(train_loader)) [0] [modality.name] 
    test_samp =  next(iter(test_loader)) [0] [modality.name] 

    model = AE(modality= modality,
               test_samp= test_samp,train_samp=train_samp,
               regularization= None, lr = lr)
    

 
    
    tb_logger =  TensorBoardLogger(save_dir = CHECKPOINT_DIR,
                                    name=str("ae_{}_{}".format(args.modality,dim_latent) )
                                    )

    trainer = pl.Trainer(
        logger= tb_logger, 
        check_val_every_n_epoch=25,
        accelerator='gpu', 
        max_epochs= NUM_epoch, 
        devices= 2,
        default_root_dir = CHECKPOINT_DIR,
          num_sanity_val_steps=0,
        strategy="ddp_find_unused_parameters_true",
      #  resume_from_checkpoint="/home/******/Documents/code/mld/trained_models/CUB/AE_mmvaeplus/ae_model_aeautokhl64_256_image0/version_2/checkpoints/epoch=299-step=82800.ckpt"
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders= test_loader) 
         #       ckpt_path="/home/******/work/mld/trained_models/CelebA/AES_1/ae_image_128/version_2/checkpoints/epoch=599-step=117000 copy.ckpt")

    