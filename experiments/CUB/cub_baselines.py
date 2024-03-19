
import sys
root = "/home/bounoua/work/mld/"
sys.path.append(root)


import os
import pytorch_lightning as pl
from src.models.MVTCAE import MVTCAE
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import logging
from pytorch_lightning import Trainer, seed_everything
from src.models.moe_mvae import MoEVAE
from src.models.poe_mvae import PoEVAE
from src.models.mopoe_mvae import MoPoEVAE
from src.models.nexus import Nexus_impl
from src.models.mmvaplus_mvae import MMVAE_plus
 
from src.dataLoaders.CUB.CUB import CubDataset
from src.dataLoaders.CUB.modalities import Sentence, Bird_Image


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
logging.getLogger("lightning").setLevel(logging.ERROR)
from src.utils import create_forlder
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', type=str, default="mopoe",
                    help=("Options possible are mvae, mmvae, mopoe,tcmvae"))

scenario_code =1


do_evaluation = False
do_fd = True
eval_epoch = 250
log_epoch = 5

limit_clip = 30000
n_fd = 10000

NUM_epoch = 150
latent_dim = 256
batch_size = 32
beta = 1.0

r_image= 32/ (64*64)
#r_sentence =1.0
r_sentence = 5.0
# r_sentence = 1.0
#lr =  1e-3
lr =  1e-3

test_batch_size = 512

# r_w_image= 1.0
# r_w_sound = 1.0
# r_w_traj = 1.0

train,test = CubDataset(train=True),CubDataset(train=False)


train_loader = DataLoader(train, batch_size=batch_size,
                          shuffle=True,
                          num_workers=4, drop_last=True,pin_memory=True)

test_loader = DataLoader(test, batch_size=test_batch_size,
                          shuffle= True,
                          num_workers=4, drop_last=True,pin_memory=True)

parser.add_argument('--seed', type=str, default=0,
                    help=("Options possible are mvae, mmvae, mopoe"))


CHECKPOINT_DIR = "trained_models/"
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'ICLR_paper_final')
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'CUB_10')

create_forlder(CHECKPOINT_DIR)
#CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'Baslines_256')


if __name__ == "__main__":
    
    
   
    #seeds =[5] 
    args = parser.parse_args()
    seeds = [int(args.seed)]
    global_results ={}
    for id_r, seed in enumerate(seeds):
        seed_everything(seed)
        
        
        modalities_list = [     Bird_Image(latent_dim=latent_dim,reconstruction_weight = r_image, lhood_name="laplace" ,resnet= "resnet" , laplace_scale =0.75 ),
                                Sentence(latent_dim=latent_dim,reconstruction_weight = r_sentence, lhood_name="categorical",resnet="resnet"),
                            ]
        #seed_everything(seed)
        if args.model =="mopoe":
            beta = 9.0
            lr =  5e-4
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
                n_fd=n_fd,
                log_epoch= log_epoch,
                lr =lr,
                dataset= "CUB"
            )
        elif args.model =="mvae":
            beta = 9.0
            lr =  5e-4
            model_mmvae = PoEVAE(
                latent_dim=latent_dim,
                batch_size =batch_size,
                beta = beta,
                limit_clip = limit_clip, 
                n_fd=n_fd,
                modalities_list =modalities_list,
                train_loader = train_loader,
                test_loader = test_loader,
                eval_epoch = eval_epoch,
                do_evaluation =do_evaluation,
                do_fd= do_fd,
                test_batch_size = test_batch_size,
                lr =lr,
                log_epoch= log_epoch, dataset= "CUB"
            )
        elif args.model =="mmvae":
            beta = 1.0
            lr =  1e-3
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
                n_fd=n_fd,       lr =lr,
                log_epoch= log_epoch, dataset= "CUB"
            )
        elif args.model == "tcmvae":
            beta = 9.0
            lr =  5e-4
            model_mmvae = MVTCAE(
            latent_dim=latent_dim,
                batch_size =batch_size,
                beta = beta,
                limit_clip = limit_clip,
                modalities_list =modalities_list,
                train_loader = train_loader,
                test_loader = test_loader,
                eval_epoch = eval_epoch,
                do_evaluation =do_evaluation,
                do_fd= do_fd,
                test_batch_size = test_batch_size,
                n_fd=n_fd,
                lr =lr,
                log_epoch= log_epoch, dataset= "CUB",
                tc_ratio= 1.0
            )

        elif args.model=="nexus":
          
          
            modalities_list = [     
                Bird_Image(latent_dim=64,reconstruction_weight = r_image, lhood_name="laplace"    ,resnet= "resnet" , laplace_scale =0.75 ),
                Sentence(latent_dim=32,reconstruction_weight = r_sentence, lhood_name="categorical",resnet=True),
                            ]
            
            model_mmvae = Nexus_impl(model_name=args.model + str(seed),
                latent_dim=64,
                batch_size =batch_size,
                beta = 9.0,
                modalities_list =modalities_list,
                train_loader = train_loader,
                test_loader = test_loader,
                eval_epoch = eval_epoch,
                do_evaluation =do_evaluation,
                do_fd = do_fd,
                max_epoch=20,
                log_epoch= log_epoch,
                test_batch_size = test_batch_size,
                dataset= "CUB",
                lr = lr,
                limit_clip = limit_clip,n_fd=n_fd
            )
        
                                        
        elif args.model =="mmvaeplus":

            latent_dim = 48
            latent_dim_w= 16
            K = 10
            elbo = "iwae"
            NUM_epoch = 50
            batch_size = 32
            r_image = 32/(64*64)
            r_sentence = 5
            beta = 1.0
            modalities_list = [ Bird_Image(latent_dim=latent_dim,latent_dim_w= latent_dim_w,reconstruction_weight = r_image,
                                            lhood_name="laplace"   
                                                ,resnet= "resnetplus" , laplace_scale =0.01 ,distengeled= True),
                                        Sentence(latent_dim=latent_dim,latent_dim_w=latent_dim_w,reconstruction_weight = r_sentence, 
                                                 lhood_name="categorical",
                                                 resnet=False,distengeled= True)
                                    ]
                
            

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
                limit_clip = limit_clip,
                n_fd=n_fd ,
                dataset= "CUB",
                log_epoch= log_epoch,
                test_batch_size = test_batch_size,
                lr = 1e-3
            )
        CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, args.model)
        tb_logger =  TensorBoardLogger(save_dir = CHECKPOINT_DIR,
                                        name=str(seed) )
        trainer = pl.Trainer(
            logger= tb_logger, 
            check_val_every_n_epoch=10,
            accelerator='gpu', 
            devices= 1 ,
            #  callbacks=[EarlyStopping(monitor="train_loss", check_finite =True), checkpoint_callback ],
            max_epochs= NUM_epoch, 
            default_root_dir = CHECKPOINT_DIR,
          # resume_from_checkpoint = "/home/bounoua/work/mld/trained_models/CUB/mvae5/version_0/checkpoints/epoch=49-step=69150.ckpt",
            num_sanity_val_steps=0,
            )
        
        trainer.fit(model=model_mmvae, train_dataloaders=train_loader, val_dataloaders= test_loader )

    