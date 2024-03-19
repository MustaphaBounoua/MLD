import sys
root = "/home/bounoua/work/mld/"
sys.path.append(root)

import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import logging
from pytorch_lightning import Trainer, seed_everything
from src.models.moe_mvae import MoEVAE
from src.models.poe_mvae import PoEVAE
from src.models.mopoe_mvae import MoPoEVAE
from src.models.MVTCAE import MVTCAE
from src.dataLoaders.MMNIST.modalities import MMNIST
from src.models.nexus import Nexus_impl
from src.models.mmvaplus_mvae import MMVAE_plus
from src.dataLoaders.MMNIST.MMNIST import get_mmnist_dataset

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
logging.getLogger("lightning").setLevel(logging.ERROR)
from src.utils import create_forlder

from datetime import date
import json

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', type=str, default="mopoe",
                    help=("Options possible are mvae, mmvae, mopoe,tcmvae,mmvaeplus"))

parser.add_argument('--seed', type=str, default=0,
                    help=(""))

eval_epoch = 300
do_evaluation = True
do_fd = True
log_epoch = 100
latent_dim =512
scenario_code = 1
test_batch_size = 512




NUM_epoch = 300
batch_size = 256
beta = 2.5
lr = 1e-3



CHECKPOINT_DIR = "trained_models/"
create_forlder(CHECKPOINT_DIR)


CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'MMNIST')

train,test = get_mmnist_dataset()

train_loader = DataLoader(train, batch_size=batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True,pin_memory =True)
test_loader = DataLoader(test, batch_size=test_batch_size,
                          shuffle= True,
                          num_workers=8, drop_last=True,pin_memory =True)


if __name__ == "__main__":
    args = parser.parse_args()
 
    #seeds = [3,4,0]
    #seeds = [2,3,4]
    seeds = [int(args.seed)]
    #seeds =[0] 
    global_results ={}
    for id_r, seed in enumerate(seeds):
        seed_everything(seed, workers=True)


        modalities_list =  [MMNIST(latent_dim=latent_dim,lhood_name="laplace",deterministic=False, 
                                   name="m{}".format(i)) for i in [0,1,2,3,4]]
        
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
                n_fd=50,
                log_epoch= log_epoch,
                lr = lr,
                 test_batch_size = test_batch_size,
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
                do_fd = do_fd,
                n_fd=50,
                log_epoch= log_epoch,
                lr = lr,   
                 test_batch_size = test_batch_size,
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
                do_fd = do_fd,
                n_fd=50,
                log_epoch= log_epoch,
                lr = lr,    test_batch_size = test_batch_size,
            )
        elif args.model =="tcmvae":
                model_mmvae = MVTCAE(
                latent_dim=latent_dim,
                batch_size =batch_size,
                beta = beta,
                modalities_list =modalities_list,
                train_loader = train_loader,
                test_loader = test_loader,
                eval_epoch = eval_epoch,
                do_evaluation =do_evaluation,
                do_fd = do_fd,
                n_fd=50,
                log_epoch= log_epoch,
                test_batch_size = test_batch_size,
                tc_ratio=5/6,
                lr = lr
            )
        
        elif args.model =="mmvaeplus":

                latent_dim = 32
                latent_dim_w= 32

                latent_dim = 32
                latent_dim_w= 32

                K = 1
                elbo = "iwae"
                beta = 2.5
                batch_size = 64
                NUM_epoch = 300
                eval_epoch = 300
                modalities_list =  [MMNIST(latent_dim=latent_dim,lhood_name="laplace",deterministic=False, 
                                        distengled= True,
                                           resnet=False,
                                           latent_dim_w = latent_dim_w, name="m{}".format(i)) for i in [0,1,2,3,4]]
                
                CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'mmvaeplus_{}_k_{}'.format(elbo,K))

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
                do_fd = do_fd,
                n_fd=5000,
                log_epoch= log_epoch,
                test_batch_size = test_batch_size,
                lr = lr
            )


        elif args.model =="nexus":
          
            modalities_list =  [MMNIST(latent_dim=160,lhood_name="laplace",deterministic=False, name="m{}".format(i)) for i in [0,1,2,3,4]]

            model_mmvae = Nexus_impl(model_name=args.model + str(seed),
                latent_dim=128,
                batch_size =batch_size,
                beta = 1.0,
                modalities_list =modalities_list,
                train_loader = train_loader,
                test_loader = test_loader,
                eval_epoch = eval_epoch,
                do_evaluation =do_evaluation,
                do_fd = do_fd,
                n_fd=5000,
                log_epoch= log_epoch,
                test_batch_size = test_batch_size,
                dataset= "MMNIST",
                max_epoch = 20,
                lr = lr
            )



        tb_logger =  TensorBoardLogger(save_dir = CHECKPOINT_DIR,
                                        name=str(args.model)+"_seed_"+str(seed),
                                        log_graph = True)

        trainer = pl.Trainer(
            logger= tb_logger,
            accelerator='gpu', devices= 1 ,
            #  callbacks=[EarlyStopping(monitor="train_loss", check_finite =True), checkpoint_callback ],
                max_epochs= NUM_epoch, 
                        num_sanity_val_steps=0,
                default_root_dir = CHECKPOINT_DIR,
          # resume_from_checkpoint = "/home/bounoua/work/mld/trained_models/MMNIST/nexus_test/nexus_seed_4/version_0/checkpoints/epoch=193-step=45396.ckpt"
                )


        
        
        
        trainer.fit(model=model_mmvae, train_dataloaders=train_loader, val_dataloaders= test_loader )

        results = model_mmvae.final_results
        results["logdir"] = model_mmvae.logdir
        global_results[seed] = results
    
