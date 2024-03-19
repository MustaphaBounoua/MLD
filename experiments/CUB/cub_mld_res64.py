

import sys
root = "/home/renault/Documents/code/mld/"
sys.path.append(root)

import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from src.utils import create_forlder
import os
from src.unimodal.BaseAE import AE
from src.models.MLD import MLD
from src.models.LateFusionAE import LateFusionAE
from pytorch_lightning import Trainer, seed_everything

from src.dataLoaders.CUB.CUB import CubDataset , Dataset_latent
from src.dataLoaders.CUB.modalities import Sentence, Bird_Image
import pickle 
from src.utils import get_stat_from_file

seed =  0

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', type=str, default="mld" + str(seed),
                    help=("Options possible are mvae, mmvae, mopoe"))
                    
#seed_everything(seed, workers=True)
parser.add_argument('--dropout', type=str, default=0.3 ,
                    help=("Options possible are "))

parser.add_argument('--seed', type=str, default=0,
                    help=("Options possible are mvae, mmvae, mopoe"))


parser.add_argument('--comment', type=str, default=0,
                    help=("Options possible are mvae, mmvae, mopoe"))

do_evaluation = True
do_fd = True

n_fd = 10000
limit_clip=30000

eval_epoch = 3000
log_epoch = 200



NUM_epoch = 3000 

batch_size = 64

r_w_image= 1.0

r_w_sentence = 1.0

lr = 1e-4


train,test = CubDataset(train=True),CubDataset(train=False)

train_latent =Dataset_latent("/home/renault/Documents/code/mld/data/cub_64/")

train_loader = DataLoader(train_latent, batch_size=batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True ,pin_memory =True)

test_loader = DataLoader(test, batch_size=batch_size,
                          shuffle= True,
                          num_workers=8, drop_last=True,pin_memory= True)



CHECKPOINT_DIR = "trained_models/"
create_forlder(CHECKPOINT_DIR)
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'cub_64')





PATHS = {
 # "image":   "/home/renault/Documents/code/mld/trained_models/CUB/AE_mmvaeplus/ae_model_full_res_512_image0/version_4/checkpoints/epoch=999-step=138000.ckpt",
 # "image":   "/home/renault/Documents/code/mld/trained_models/CUB/AE_mmvaeplus/ae_model_transf_256_image0/version_3/checkpoints/epoch=999-step=276000.ckpt",
   "image" :"/home/renault/Documents/code/mld/trained_models/cub_64/image_64/version_3/checkpoints/epoch=499-step=69000.ckpt"
    ,"sentence":"/home/renault/Documents/code/mld/trained_models/CUB/paper_ae/text_resnet/epoch=1009-step=1396830.ckpt" }




if __name__ == "__main__":
    
    args = parser.parse_args()
    
    dim_image = 64
    dim_sentence = 32
 
 
   # seed_everything(int(args.seed))
    modalities_list = [     Bird_Image(h=64,latent_dim=dim_image,reconstruction_weight = r_w_image, lhood_name="laplace" ,deterministic=True,resnet = "autokl"),
                            Sentence(latent_dim=dim_sentence,reconstruction_weight = r_w_sentence, lhood_name="categorical",deterministic=True,resnet = "resnet"),
                         ]

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
                nb_samples = 6,
                log_epoch= log_epoch ,
                lr = lr,
                nb_batchs= None,
                do_class = False,
                time_dim= 512,
                unet_architecture = (1,1),
                unet_type='linear',
                dropout=float(args.dropout),
                init_dim= 512*3,
                preprocess_type = "modality",
                preprocess_op = "standerdize",
                check_stat = False,
                betas=[0.1,20],
                train_batch_size = batch_size,
                N_step= 250, 
                importance_sampling= True,
                ll_weight= False,
                group_norm=32,
                debug = False,
                use_attention = False,
                shift_scale = False ,
                num_head = 1   ,
                use_ema=True,
                cross_gen="repaint",
                dataset="CUB",
                n_fd=n_fd,
                limit_clip = limit_clip ,train_latent= True,
            )
        
    CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'MLDautokl_latent_64res_imp'+str(args.dropout))
    #  print(model)
    tb_logger =  TensorBoardLogger(save_dir = CHECKPOINT_DIR,
                                    name=str(args.seed)
                                    )
    #model.stat = get_stat_from_file("/home/bounoua/work/mld/trained_models/CUB/MMLD_f/mmld0/version_0/stat.pickle")
    
    trainer = pl.Trainer(
        logger = tb_logger, 
        check_val_every_n_epoch=50,
        accelerator ='gpu', 
        devices = 2   ,
        max_epochs= NUM_epoch, 
        default_root_dir = CHECKPOINT_DIR,
        num_sanity_val_steps=0,
        strategy="ddp",
     #   deterministic= True,
        #   resume_from_checkpoint = "/home/bounoua/work/mld/trained_models/CUB/MMLD_f/mmld0/version_0/checkpoints/epoch=1299-step=448500.ckpt"
            )

    
    trainer.fit(model=model, train_dataloaders=model.train_loader, val_dataloaders= model.test_loader )
    