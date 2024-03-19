import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from src.dataLoaders.MHD.MHD import MHDDataset
from src.dataLoaders.MHD.modalities import Image_mod,Sound,Trajectory
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from src.utils import create_forlder
import os
from src.unimodal.BaseAE import AE
from src.models.MLD import MLD
from src.models.LateFusionAE import LateFusionAE
from pytorch_lightning import Trainer, seed_everything

seed =  1

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', type=str, default="mmld_seed" + str(seed),
                    help=("Options possible are mvae, mmvae, mopoe"))
                    
#seed_everything(seed, workers=True)



parser.add_argument('--seed', type=str, default=0,
                    help=("Options possible are mvae, mmvae, mopoe"))


do_evaluation = False
do_fd = True
eval_epoch = 500
log_epoch = 3000



NUM_epoch = 3000  
batch_size = 128
r_w_image= 1.0
r_w_sound = 1.0
r_w_traj = 1.0
lr = 1e-4

train,test = MHDDataset(train=True),MHDDataset(train=False)


train_loader = DataLoader(train, batch_size=batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True)

test_loader = DataLoader(test, batch_size=batch_size,
                          shuffle= True,
                          num_workers=4, drop_last=True)



CHECKPOINT_DIR = "trained_models/"
create_forlder(CHECKPOINT_DIR)
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'MHD')




PATHS = {
    "image":"trained_models/MHD/AE/ae_deterimage/version_1/checkpoints/epoch=0-step=781.ckpt",
    "sound":"trained_models/MHD/AE/ae_detersound/version_1/checkpoints/epoch=0-step=781.ckpt",
    "trajectory":"trained_models/MHD/AE/ae_detertrajectory/version_1/checkpoints/epoch=0-step=781.ckpt"
    }
    
if __name__ == "__main__":
    
    args = parser.parse_args()
    CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'MMLD_08')
    seeds =[int(args.seed)] 
    dim_sound = 128
    dim_image = 64
    dim_traj = 16 
    for seed in seeds:
        seed_everything(seed)
        modalities_list = [     Image_mod(latent_dim=dim_image,reconstruction_weight = r_w_image, deterministic = True ),
                                Sound(latent_dim=dim_sound,reconstruction_weight = r_w_sound, pretrained = False,deterministic =True),
                                Trajectory(latent_dim=dim_traj,reconstruction_weight = r_w_traj,deterministic = True ) ]
    
        aes = []
        for mod in modalities_list:
            print(mod.name)
            aes.append(
                AE.load_from_checkpoint( PATHS[mod.name] , modality = mod).eval()
            )
        aes_model = LateFusionAE( aes = aes )
        aes_model.eval()
            
        
        

        model = MLD(
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
                    nb_batchs= 10,
                    do_class = False,
                    time_dim= 512,
                    unet_architecture = (1,1),
                    unet_type='linear',
                    d= 0.8,
                    init_dim= 1024,
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
                    use_ema=True,
                    cross_gen="repaint",
            dataset = "MHD"
                )
            

        print(model)
        tb_logger =  TensorBoardLogger(save_dir = CHECKPOINT_DIR,
                                        name=str(seed)
                                        )

        trainer = pl.Trainer(
            logger = tb_logger, 
            check_val_every_n_epoch=100,
            accelerator ='gpu', 
            devices = 1   ,
            max_epochs= NUM_epoch, 
            default_root_dir = CHECKPOINT_DIR,
            num_sanity_val_steps=0,

                )

        
        trainer.fit(model=model, train_dataloaders=model.train_loader, val_dataloaders= model.test_loader )
        