
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from src.dataLoaders.MHD.MHD import MHDDataset
from src.dataLoaders.MHD.modalities import Image_mod,Sound,Trajectory
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from src.utils import create_forlder
import os
from src.utils import get_stat_from_file
from src.models.MLD import MLD
from src.unimodal_vae.BaseAE import AE
from src.models.LateFusionAE import LateFusionAE
from pytorch_lightning import Trainer, seed_everything
from src.dataLoaders.celebA.CelebA import CelebAHQMaskDS ,Dataset_latent ,Dataset_latent_2
from src.dataLoaders.celebA.modalities import Image_mod, Mask_mod, Attributes

seed =  1

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', type=str, default="mmld_seed" + str(seed),
                    help=("Options possible are mvae, mmvae, mopoe"))




parser.add_argument('--seed', type=str, default=1,
                    help=("Options possible are mvae, mmvae, mopoe"))


do_evaluation = False
do_fd = False
eval_epoch = 500
log_epoch = 250



NUM_epoch = 3000  
batch_size = 64
lr = 1e-4


train,test = Dataset_latent_2(train=True ,im=256),Dataset_latent_2(train=False,im=256)

Dataset_latent
train_loader = DataLoader(train, batch_size=batch_size,
                          shuffle=True,pin_memory=True,
                          num_workers=16, drop_last=True)

test_loader = DataLoader(test, batch_size=batch_size,
                          shuffle= True,pin_memory=True,
                          num_workers=16, drop_last=True)



CHECKPOINT_DIR = "trained_models/"
create_forlder(CHECKPOINT_DIR)
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'CelebA')




PATHS = { 
      "image":"/home/******/work/mld/trained_models/CelebA/AES_1/ae_image_256/version_0/checkpoints/epoch=299-step=58500 copy.ckpt",
    "mask":"/home/******/work/mld/trained_models/CelebA/AES_1/ae_mask_128/version_3/checkpoints/epoch=199-step=39000.ckpt",
    "attributes":"/home/******/work/mld/trained_models/CelebA/AES_1/ae_att_32/version_0/checkpoints/epoch=99-step=39000.ckpt"
    }
    
if __name__ == "__main__":
    
    args = parser.parse_args()
    CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'MLD_256_128_32')
    seeds =[int(args.seed)] 
    
    dim_image = 256
    dim_mask = 128
    dim_att = 32 
    

    seed_everything(seed)
    modalities_list = [   Image_mod(latent_dim=dim_image ),
                           Mask_mod(latent_dim=dim_mask ) ,
                           Attributes(latent_dim=dim_att  )   ]
    
    aes = []
    for mod in modalities_list:
            print(mod.name)
            aes.append(
                AE.load_from_checkpoint( PATHS[mod.name] , modality = mod).eval()
            )
    aes_model = LateFusionAE( aes = aes )
    aes_model.eval()
            
        
        

    model = MLD(    aes= aes_model,
                    batch_size =batch_size,
                    train_loader = train_loader,
                    test_loader = test_loader,
                    eval_epoch = eval_epoch,
                    do_evaluation =do_evaluation,
                    do_fd = do_fd,
                    nb_samples = 3,
                    n_fd=2500,
                    nb_batchs= 40,
                    log_epoch= log_epoch ,
                    lr = lr,
                    do_class = False,
                    time_dim= 512,
                    unet_architecture = (1,1),
                    unet_type='linear',
                    dropout= 0.5,
                    init_dim= 512*3,
                    preprocess_type = "modality",
                    preprocess_op = "standerdize",
                    check_stat = False,
                    # preprocess_type = None,
                    # preprocess_op = None,
                    # check_stat = False,
                    betas=[0.1,20],
                    train_batch_size = batch_size,
                    N_step= 250, 
                    importance_sampling= False,
                    ll_weight= False,
                    group_norm=32,
                    debug = False,
                    use_attention = False,
                    shift_scale = False ,
                    num_head = 1   ,
                    train_latent= True,
                    use_ema=True,
                    cross_gen="repaint",
            dataset = "celebA"
                )


    tb_logger =  TensorBoardLogger(save_dir = CHECKPOINT_DIR,
                                        name=str(seed)
                                        )

    trainer = pl.Trainer(
            logger = tb_logger, 
            check_val_every_n_epoch=100,
            accelerator ='gpu', 
            devices = 4,
            max_epochs= NUM_epoch, 
            default_root_dir = CHECKPOINT_DIR,
            num_sanity_val_steps=0,
            strategy="ddp_find_unused_parameters_true",
        #   deterministic= True,
        #    resume_from_checkpoint = "trained_models/MNISTSVHN/mld/version_30/checkpoints/epoch=49-step=876050.ckpt"
                )

        
    trainer.fit(model=model, 
                train_dataloaders=model.train_loader, 
                val_dataloaders= model.test_loader )
        