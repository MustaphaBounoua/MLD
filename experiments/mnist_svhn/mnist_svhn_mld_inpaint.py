
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything

from src.dataLoaders.MnistSvhnText.MnistSvhnText import  get_data_set_svhn_mnist

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from src.utils import create_forlder,get_stat
import os

from src.unimodal.BaseAE import AE
from src.models.MLD_inpaint import MLD_Inpaint
from src.models.LateFusionAE import LateFusionAE
from src.dataLoaders.MnistSvhnText.modalities import MNIST,SVHN,LABEL
import pickle
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)


parser.add_argument('--model', type=str, default="mmld_repaint",
                    help=("Options possible are mvae, mmvae, mopoe"))


parser.add_argument('--seed', type=str, default= 0 ,
                    help=("Options possible are "))

do_evaluation = False
do_fd = True
do_class = False
eval_epoch = 15
log_epoch = 50



NUM_epoch = 150
batch_size = 128

r_w_mnist= 1.0
r_w_svhn = 1.0
r_w_text = 1.0
lr = 1e-4

PATHS = {
    "mnist": "trained_models/MNISTSVHN/aemnist/version_2/checkpoints/epoch=3-step=35040.ckpt",
    "svhn": "trained_models/MNISTSVHN/aesvhn/version_0/checkpoints/epoch=10-step=96360.ckpt"}





train,test = get_data_set_svhn_mnist(with_text= False)

train_loader = DataLoader(train, batch_size=batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True,pin_memory= True)

test_loader = DataLoader(test, batch_size=512,
                          shuffle= True,
                          num_workers=4, drop_last=True,pin_memory= True)


CHECKPOINT_DIR = "trained_models/"
create_forlder(CHECKPOINT_DIR)
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'MNISTSVHN')


if __name__ == "__main__":
    
    args = parser.parse_args()
    
    dim_mnist = 16
    dim_svhn = 64
    dim_label = 10 

    modalities_list =  [
        MNIST(latent_dim=dim_mnist, reconstruction_weight= r_w_mnist, deterministic = True, lhood_name="laplace") , 
        SVHN(latent_dim=dim_svhn, reconstruction_weight=r_w_svhn, deterministic = True, lhood_name="laplace" )  
    ]
    aes = []
    for mod in modalities_list:
        aes.append(
            AE.load_from_checkpoint( PATHS[mod.name] , modality = mod)
        )
    aes_model = LateFusionAE( aes = aes )
    aes_model.eval()
    
    
    
    seeds = [int(args.seed)]

    for seed in seeds:
        
        seed_everything(seed)
        
        model = MLD_Inpaint(
                    aes = aes_model,
                    model_name = "mmld",
                    batch_size =batch_size,
                    train_loader = train_loader,
                    test_loader = test_loader,
                    eval_epoch = eval_epoch,
                    do_evaluation = do_evaluation,
                    do_fd = do_fd,
                    nb_samples = 8,
                    n_fd=5000,
                    log_epoch= log_epoch ,
                    lr = lr,
                    nb_batchs = None,
                    init_dim= 512,
                    do_class = False,
                    time_dim =  256,
                    unet_architecture = (1,1) ,
                    unet_type='linear',
                    d= 0,
                    preprocess_type = "modality",
                    preprocess_op = "standerdize",
                    check_stat = False,
                    group_norm = 8,
                    # preprocess_type = None,
                    # preprocess_op = None,
                    # check_stat = False,
                    betas=[0.1,20],
                    train_batch_size = 512,
                    N_step= 250, 
                    importance_sampling= False,
                    ll_weight= False,
                    debug = False,
                    use_attention = False,
                    shift_scale = False,
                    num_head = 3,
                    use_ema= True,
                    cross_gen= "repaint"
                )
        CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, str(args.model))
        tb_logger =  TensorBoardLogger(save_dir = CHECKPOINT_DIR,
                                        name=  str(args.seed) )
      #  model.stat= get_stat("/home/bounoua/work/mld/trained_models/MNISTSVHN/mmld_repaint/3/version_0/stat.pickle")
        
        trainer = pl.Trainer(
            logger = tb_logger, 
            check_val_every_n_epoch=50,
            accelerator='gpu', 
            devices= 1    ,
            max_epochs= NUM_epoch, 
            default_root_dir = CHECKPOINT_DIR,
            num_sanity_val_steps=0,
            deterministic= True,
          #     resume_from_checkpoint= "/home/bounoua/work/mld/trained_models/MNISTSVHN/mmld_repaint/3/version_0/checkpoints/epoch=139-step=1226400.ckpt"
                )
        


        trainer.fit(model=model, train_dataloaders=model.train_loader, val_dataloaders= model.test_loader )
    