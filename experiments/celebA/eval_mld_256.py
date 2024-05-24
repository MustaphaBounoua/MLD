
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from src.utils import create_forlder
import os
from src.utils import get_stat_from_file,get_root_folder
from src.models.MLD import MLD
from src.unimodal.BaseAE import AE
from src.models.LateFusionAE import LateFusionAE
from pytorch_lightning import Trainer, seed_everything
from src.dataLoaders.celebA.CelebA import CelebAHQMaskDS ,Dataset_latent
from src.dataLoaders.celebA.modalities import Image_mod, Mask_mod, Attributes
import json
seed =  1

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', type=str, default="mmld_seed" + str(seed),
                    help=("Options possible are mvae, mmvae, mopoe"))
                    
#seed_everything(seed, workers=True)



parser.add_argument('--seed', type=str, default=0,
                    help=("Options possible are mvae, mmvae, mopoe"))


do_evaluation = False
do_fd = False
eval_epoch = 500
log_epoch = 100



NUM_epoch = 3000  
batch_size = 64
lr = 1e-4

train,test = CelebAHQMaskDS(train=True, all_mod ="all" ),CelebAHQMaskDS(train=False,all_mod ="all")


Dataset_latent
train_loader = DataLoader(train, batch_size=batch_size,
                          shuffle=True,pin_memory=True,
                          num_workers=8, drop_last=True)

test_loader = DataLoader(test, batch_size=256,
                          shuffle= True,pin_memory=True,
                          num_workers=8, drop_last=True)



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
    CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'MLD')
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
            
        
        
    PATH="/home/******/work/mld/trained_models/CelebA/MLD_256_128_32/1/version_6/checkpoints/epoch=2999-step=291000.ckpt"
    model = MLD.load_from_checkpoint(PATH    ,aes= aes_model,
                    batch_size =batch_size,
                    train_loader = train_loader,
                    test_loader = test_loader,
                    eval_epoch = eval_epoch,
                    do_evaluation =do_evaluation,
                    do_fd = do_fd,
                    nb_samples = 3,
                    n_fd=5000,
                    nb_batchs= None,
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
                    train_batch_size = 256,
                    N_step= 250, 
                    importance_sampling= False,
                    ll_weight= False,
                    group_norm=32,
                    debug = False,
                    use_attention = False,
                    shift_scale = False ,
                    num_head = 1   ,
                    train_latent= False,
                    use_ema=True,
                    cross_gen="repaint",
            dataset = "celebA"
                )
    
    model.stat = get_stat_from_file("/home/******/work/mld/trained_models/CelebA/MLD_256_128_32/1/version_6/stat.pickle")
    
    model = model.to("cuda:0")
    model.sde.device = model.device
    model = model.eval()
   
    model.do_fd = True
    model.n_fd = 5000
    model.train_batch_size = 256
    model.nb_batchs = None

    results = model.evaluation()
    print(results)
    with open(os.path.join(get_root_folder(PATH),"results_all.json"),"w") as f:
              json.dump(results,f)