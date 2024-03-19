

import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

import os
import pickle
from src.dataLoaders.MnistSvhnText.MnistSvhnText import  get_data_set_svhn_mnist

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from src.utils import create_forlder
import os

from src.unimodal.BaseAE import AE
from src.models.MLD import MLD
from src.models.LateFusionAE import LateFusionAE
from src.dataLoaders.MnistSvhnText.modalities import MNIST,SVHN,LABEL
from src.dataLoaders.MHD.modalities import Image_mod,Sound,Trajectory
from src.dataLoaders.MHD.MHD import MHDDataset

from src.dataLoaders.MMNIST.modalities import MMNIST
from src.dataLoaders.MMNIST.MMNIST import get_mmnist_dataset

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', type=str, default="ms",
                    help=("Options possible are mvae, mmvae, mopoe"))




do_evaluation = True
do_fd = True

batch_size = 256





CHECKPOINT_DIR = "trained_models/"
create_forlder(CHECKPOINT_DIR)
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'AE_exp')


if __name__ == "__main__":
    
    args = parser.parse_args()
    if args.dataset == "ms":
        train,test = get_data_set_svhn_mnist(with_text= False)

        train_loader = DataLoader(train, batch_size=512,
                                  shuffle=True,
                                  num_workers=0, drop_last=True)

        test_loader = DataLoader(test, batch_size=512,
                                  shuffle= True,
                                  num_workers=8, drop_last=True)
        dim_mnist = 16
        dim_svhn = 64
        PATHS = {
  "mnist":"",
   "svhn":"" }


        modalities_list =  [
            MNIST(latent_dim=dim_mnist, reconstruction_weight= 1, deterministic = True, lhood_name="laplace"),
            SVHN(latent_dim=dim_svhn, reconstruction_weight=1, deterministic = True, lhood_name="laplace" ,v2 = '3')  
        ]
        
    elif args.dataset=="mhd":
        train,test = MHDDataset(train=True),MHDDataset(train=False)


        train_loader = DataLoader(train, batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=1, drop_last=True)

        test_loader = DataLoader(test, batch_size=512,
                                  shuffle= True,
                                  num_workers=1, drop_last=True)
        PATHS = {
    "image":"",
    "sound":"",
    "trajectory":""
    }
        modalities_list = [     Image_mod(latent_dim=64,reconstruction_weight = 1, deterministic = True ),
                            Sound(latent_dim=128,reconstruction_weight = 1, pretrained = False,deterministic =True),
                            Trajectory(latent_dim=16,reconstruction_weight = 1,deterministic = True ) ]
   
    elif args.dataset=="mmnist" :
        train,test = get_mmnist_dataset()

        train_loader = DataLoader(train, batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=8, drop_last=True)
        test_loader = DataLoader(test, batch_size=512,
                                  shuffle= True,
                                  num_workers=8, drop_last=True)
        PATHS = {
     "m0":"trained_models/MMNIST/ae_deter_160_m0/version_0/checkpoints/epoch=299-step=281100.ckpt",
     "m1":"trained_models/MMNIST/ae_deter_160_m1/version_0/checkpoints/epoch=299-step=281100.ckpt" ,
     "m2":"trained_models/MMNIST/ae_deter_160_m2/version_0/checkpoints/epoch=299-step=281100.ckpt",
     "m3":"trained_models/MMNIST/ae_deter_160_m3/version_0/checkpoints/epoch=299-step=281100.ckpt",
     "m4":"trained_models/MMNIST/ae_deter_160_m4/version_0/checkpoints/epoch=299-step=281100.ckpt"
}


        latent_dim = 160

 
    
        modalities_list =  [MMNIST(latent_dim=latent_dim,lhood_name="laplace",deterministic=True, name="m{}".format(i)) for i in [0,1,2,3,4] ]
  
    aes = []
    for mod in modalities_list:
            aes.append(
                AE.load_from_checkpoint( PATHS[mod.name] , modality = mod)
            )
    
    aes_model = LateFusionAE( aes = aes , train_loader= train_loader, test_loader= test_loader)

    tb_logger =  TensorBoardLogger(save_dir = CHECKPOINT_DIR,
                                        name=str(args.dataset)
                                        )
    print(aes_model)

    trainer = pl.Trainer(
            logger = tb_logger, 
            check_val_every_n_epoch=5,
            accelerator='gpu', 
            devices=  1 ,
            max_epochs= 1, 
            default_root_dir = CHECKPOINT_DIR,
            num_sanity_val_steps=0,
           # resume_from_checkpoint="trained_models/MNISTSVHN/mmld/version_11/checkpoints/epoch=49-step=219000.ckpt"
                )
    trainer.fit(model=aes_model, train_dataloaders=aes_model.train_loader, val_dataloaders= aes_model.test_loader )
    