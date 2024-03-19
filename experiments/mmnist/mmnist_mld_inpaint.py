import sys
root = "/home/bounoua/work/mld/"
sys.path.append(root)

import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from src.utils import create_forlder
import os
from src.unimodal.BaseAE import AE
from src.models.MLD_inpaint import MLD_Inpaint
from src.models.LateFusionAE import LateFusionAE
from pytorch_lightning import Trainer, seed_everything
import pickle
from src.dataLoaders.MMNIST.modalities import MMNIST
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from src.dataLoaders.MMNIST.MMNIST import get_mmnist_dataset
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', type=str, default="mld_inpaint",
                    help=("Options possible are mvae, mmvae, mopoe"))

parser.add_argument('--seed', type=str, default=0,
                    help=("Options possible are mvae, mmvae, mopoe"))
                    
    
#seed_everything(seed, workers=True)


do_evaluation = False
do_fd = True
eval_epoch = 1000
log_epoch = 1000



NUM_epoch = 3000  
batch_size = 256

lr = 1e-4


test_batch_size = 512


CHECKPOINT_DIR = "trained_models/"
create_forlder(CHECKPOINT_DIR)
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'MMNIST')




PATHS = {
    "m0": "trained_models/MMNIST/ae_deter_latentm0/version_1/checkpoints/epoch=0-step=468.ckpt",
    "m1": "trained_models/MMNIST/ae_deter_latentm0/version_1/checkpoints/epoch=0-step=468.ckpt",
    "m2": "trained_models/MMNIST/ae_deter_latentm0/version_1/checkpoints/epoch=0-step=468.ckpt",
    "m3": "trained_models/MMNIST/ae_deter_latentm0/version_1/checkpoints/epoch=0-step=468.ckpt",
    "m4": "trained_models/MMNIST/ae_deter_latentm0/version_1/checkpoints/epoch=0-step=468.ckpt",
}




train,test = get_mmnist_dataset()

train_loader = DataLoader(train, batch_size=batch_size,
                          shuffle=True,
                          num_workers=16, drop_last=True,pin_memory=True)
test_loader = DataLoader(test, batch_size=test_batch_size,
                          shuffle= True,
                          num_workers=16, drop_last=True,pin_memory=True)

def get_stat(file_name):
    t = pickle.load(open(file_name, "rb"))
    for key in t.keys():
        if  key!= 'cat':
            t[key]['mean'] = t[key]['mean'].to("cuda")
            t[key]['std'] = t[key]['std'].to("cuda")
    return t

if __name__ == "__main__":
    
    args = parser.parse_args()
    seeds = [0,1,2,3,4]
    seeds = [int(args.seed )]
    #latent_dim = 128
    latent_dim = 160
  
    CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, str(args.model))
    #seed_everything(seed, workers=True)
    modalities_list =  [MMNIST(latent_dim=latent_dim,lhood_name="laplace",deterministic=True, name="m{}".format(i)) for i in [0,1,2,3,4] ]
  
    aes = []
    for mod in modalities_list:
        aes.append(
            AE.load_from_checkpoint( PATHS[mod.name] , modality = mod)
        )
    aes_model = LateFusionAE( aes = aes )
    aes_model.eval()
    
    
    for seed in seeds:
        seed_everything(seed)
        model = MLD_Inpaint(model_name ="mld",
                    aes= aes_model,
                    batch_size =batch_size,
                    train_loader = train_loader,
                    test_loader = test_loader,
                    eval_epoch = eval_epoch,
                    do_evaluation =do_evaluation,
                    do_fd = do_fd,
                    nb_samples = 3,
                    n_fd=5000,
                    log_epoch= log_epoch ,
                    lr = lr,
                    nb_batchs= None,
                    do_class = False,
                    init_dim = 512*3,
                    time_dim= 512,
                    unet_architecture = (1,1),
                    unet_type='linear',
                    preprocess_type = "modality",
                    preprocess_op = "standerdize",
                    check_stat = False,
                    betas=[0.1,20],
                    train_batch_size = test_batch_size,
                    N_step= 250, 
                    importance_sampling= False,
                    ll_weight= False,
                    group_norm=32,
                    debug = False,
                    use_attention = False,
                    shift_scale = False ,
                    num_head = 1   ,
                    use_ema=True,
                    cross_gen="repaint"
                )

        print(model)
      #  model.stat = get_stat("/home/bounoua/work/mld/trained_models/MMNIST/mld_1t_inpaint/3_seed/version_0/stat.pickle")
        tb_logger =  TensorBoardLogger(save_dir = CHECKPOINT_DIR,
                                        name=str(args.seed)+"_seed"
                                        )

        trainer = pl.Trainer(
            logger = tb_logger, 
            check_val_every_n_epoch=25,
            accelerator ='gpu', 
            devices = 1   ,
            max_epochs= NUM_epoch, 
            default_root_dir = CHECKPOINT_DIR,
            num_sanity_val_steps=0,
         #   deterministic= True,
        #      resume_from_checkpoint = "/home/bounoua/work/mld/trained_models/MMNIST/mld_1t_inpaint/3_seed/version_0/checkpoints/epoch=374-step=87750.ckpt"
                )


        trainer.fit(model=model, train_dataloaders=model.train_loader, val_dataloaders= model.test_loader )
