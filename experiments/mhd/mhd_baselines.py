

import os
import pytorch_lightning as pl
from src.models.MVTCAE import MVTCAE
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import logging

from src.models.moe_mvae import MoEVAE
from src.models.poe_mvae import PoEVAE
from src.models.mopoe_mvae import MoPoEVAE
from src.models.nexus import Nexus_impl
from src.models.mmvaplus_mvae import MMVAE_plus
from pytorch_lightning import Trainer, seed_everything
from src.dataLoaders.MHD.MHD import MHDDataset
from src.dataLoaders.MHD.modalities import Image_mod,Sound,Trajectory

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
logging.getLogger("lightning").setLevel(logging.ERROR)
from src.utils import create_forlder
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from datetime import date
import json

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', type=str, default="mopoe",
                    help=("Options possible are mvae, mmvae, mopoe,tcmvae"))
parser.add_argument('--seed', type=int, default=1)

scenario_code =1


do_evaluation = True
do_fd = True
eval_epoch = 600
log_epoch = 50


# NUM_epoch = 600
# latent_dim = 128
# batch_size = 128
# beta = 1.0
# r_w_image= (32*128) / (28*28)
# r_w_sound = 1.0
# r_w_traj = (32*128) / 200
# lr = 0.001

test_batch_size = 256
# r_w_image= 1.0
# r_w_sound = 1.0
# r_w_traj = 1.0


NUM_epoch = 600
latent_dim = 128
batch_size = 64
beta = 1.0
r_w_image= 5.224489795918367
r_w_sound = 1.0
r_w_traj =  20.48
#r_w_image= 1.0
# r_w_sound = 1.0
# r_w_traj = 50.0

# r_w_traj = 50.0
lr = 0.001


train,test = MHDDataset(train=True),MHDDataset(train=False)


train_loader = DataLoader(train, batch_size=batch_size,
                          shuffle=True,
                          num_workers=4, drop_last=True, pin_memory=True)

test_loader = DataLoader(test, batch_size=test_batch_size,
                          shuffle= True,
                          num_workers=4, drop_last=True,pin_memory=True)




CHECKPOINT_DIR = "trained_models/"
create_forlder(CHECKPOINT_DIR)
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'MHD')

# def seed_everything(seed):
#     np.random_seed(seed)
#     torch.manual_seed(seed)
#     random.seed(seed)

if __name__ == "__main__":
    seeds = [0,1,2,3,4]
    
    #  seeds =[4] 
    args = parser.parse_args()
    seeds = [args.seed]
    global_results ={}
    for id_r, seed in enumerate(seeds):
        seed_everything(seed)
        
    
        modalities_list = [     Image_mod(latent_dim=latent_dim,reconstruction_weight = r_w_image ),
                            Sound(latent_dim=latent_dim,reconstruction_weight = r_w_sound, pretrained = False),
                            Trajectory(latent_dim=latent_dim,reconstruction_weight = r_w_traj ) ]
    
        if args.model =="mopoe":
        
            model_mmvae = MoPoEVAE(
                latent_dim=latent_dim,
                batch_size =batch_size,
                beta = beta,
                modalities_list =modalities_list,
                train_loader = train_loader,
                test_loader = test_loader,
                eval_epoch = eval_epoch,
                do_evaluation =do_evaluation,
                do_fd = do_fd,
                test_batch_size = test_batch_size,
                n_fd=50,
                log_epoch= log_epoch,
            )
        elif args.model =="mvae":
        
            model_mmvae = PoEVAE(
                latent_dim=latent_dim,
                batch_size =batch_size,
                beta = beta,
                modalities_list =modalities_list,
                train_loader = train_loader,
                test_loader = test_loader,
                eval_epoch = eval_epoch,
                do_evaluation =do_evaluation,
                do_fd= do_fd,nb_batchs=None,
                test_batch_size = test_batch_size,
                n_fd=5000,
                log_epoch= log_epoch,
            )
        elif args.model =="mmvae":
        
            model_mmvae = MoEVAE(
            latent_dim=latent_dim,
                batch_size =batch_size,
                beta = beta,
                modalities_list =modalities_list,
                train_loader = train_loader,
                test_loader = test_loader,
                eval_epoch = eval_epoch,
                do_evaluation =do_evaluation,
                do_fd= do_fd,
                test_batch_size = test_batch_size,
                n_fd=50,
                log_epoch= log_epoch,
            )
        elif args.model == "tcmvae":
            
            model_mmvae = MVTCAE(
            latent_dim=latent_dim,
                batch_size =batch_size,
                beta = beta,
                modalities_list =modalities_list,
                train_loader = train_loader,
                test_loader = test_loader,
                eval_epoch = eval_epoch,
                nb_batchs= 10,
                do_evaluation =do_evaluation,
                do_fd= do_fd,
                test_batch_size = test_batch_size,
                n_fd=5000,
                log_epoch= log_epoch,
                tc_ratio= 5/6
            )
        elif args.model=="nexus":
          
          
            modalities_list = [     Image_mod(latent_dim=64,reconstruction_weight = 1.0 ),
                            Sound(latent_dim=128,reconstruction_weight = 1.0,pretrained= True),
                            Trajectory(latent_dim=16,reconstruction_weight = 50.0 ) ]
    
            
            model_mmvae = Nexus_impl(model_name=args.model + str(seed),
                latent_dim=32,
                batch_size =batch_size,
                beta = 1,
                modalities_list =modalities_list,
                train_loader = train_loader,
                test_loader = test_loader,
                eval_epoch = eval_epoch,
                do_evaluation =do_evaluation,
                do_fd = do_fd,
                n_fd=50,
                max_epoch=20,
                log_epoch= log_epoch,
                test_batch_size = test_batch_size,
                dataset= "MHD",
                lr = lr
            )
        elif args.model =="mmvaeplus":
            NUM_epoch = 50
            latent_dim = 20
            latent_dim_w= 64
            beta = 1.0
            K = 10
            batch_size = 32
            lr = 1e-3
            elbo = "iwae"
            modalities_list = [     Image_mod(latent_dim=latent_dim,reconstruction_weight = r_w_image,distengled=True,
                                              latent_dim_w=latent_dim_w ,lhood_name="laplace"),
                            Sound(latent_dim=latent_dim,reconstruction_weight = r_w_sound, pretrained = False,
                                  lhood_name="laplace",
                                  distengled=True,latent_dim_w=latent_dim_w),
                            Trajectory(latent_dim=latent_dim,
                            reconstruction_weight = r_w_traj,
                                       lhood_name="laplace",
                                       distengled=True,latent_dim_w=latent_dim_w ) ]
    
                
            CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'final_mmvaeplus_{}_k_{}'.format(elbo,K))

            model_mmvae = MMVAE_plus(
                latent_dim=latent_dim,
                latent_dim_w= latent_dim_w,
                batch_size =batch_size,
                beta = beta, 
                K=K,
                elbo=elbo,
                modalities_list =modalities_list,
                train_loader = train_loader,
                test_loader = test_loader,
                eval_epoch = eval_epoch,
                do_evaluation =do_evaluation,
                do_fd = False,
                n_fd=5000,
                nb_batchs=10,
                log_epoch= log_epoch,
                test_batch_size = test_batch_size,
                lr = lr
            )
        tb_logger =  TensorBoardLogger(save_dir = CHECKPOINT_DIR,
                                    name=str(seed)
                                    )

        trainer = pl.Trainer(
        logger= tb_logger, 
        check_val_every_n_epoch=100,
        accelerator='gpu', 
        devices= 1 ,
        #  callbacks=[EarlyStopping(monitor="train_loss", check_finite =True), checkpoint_callback ],
            max_epochs= NUM_epoch, 
            default_root_dir = CHECKPOINT_DIR,
        #       resume_from_checkpoint="trained_models/MHD/tcmvae_seed_0/version_0/checkpoints/epoch=799-step=624800.ckpt"
        #    deterministic= True
            )

        trainer.fit(model=model_mmvae, train_dataloaders=train_loader, val_dataloaders= test_loader
                    )

        results = model_mmvae.final_results
        results["logdir"] = model_mmvae.logdir
        global_results[seed] = results
    
  #  global_results["summary"] = get_mean_var(results)
    today = date.today()

    with open(os.path.join("trained_models","results_mhd_{}_{}.json".format(args.model,str(today))),"w") as f:
        json.dump(global_results,f)