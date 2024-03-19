


from src.dataLoaders.MnistSvhnText.modalities import MNIST, SVHN, LABEL
from src.utils import create_forlder
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from src.dataLoaders.MnistSvhnText.MnistSvhnText import get_data_set_svhn_mnist
from src.models.nexus import Nexus_impl
from src.models.MVTCAE import MVTCAE
from src.models.mopoe_mvae import MoPoEVAE
from src.models.poe_mvae import PoEVAE
from src.models.moe_mvae import MoEVAE
from src.models.mmvaplus_mvae import MMVAE_plus
from pytorch_lightning import Trainer, seed_everything
import logging
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os


logging.getLogger("lightning").setLevel(logging.ERROR)


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', type=str, default="mopoe",
                    help=("Options possible are mvae, mmvae, mopoe,tcmvae, nexus"))

parser.add_argument('--seed', type=int, default=0)



eval_epoch = 75
do_evaluation = True
do_fd = True
log_epoch = 50
do_classification = False

NUM_epoch = 150
latent_dim = 20

test_batch_size = 256
beta = 2.5
batch_size = 256
r_w_mnist = 3.92
r_w_svhn = 1.0
lr = 0.001

train, test = get_data_set_svhn_mnist(with_text=False)

train_loader = DataLoader(train, batch_size=batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True)

test_loader = DataLoader(test, batch_size=test_batch_size,
                         shuffle=True,
                         num_workers=8, drop_last=True)


CHECKPOINT_DIR = "trained_models/"
create_forlder(CHECKPOINT_DIR)


CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'MNISTSVHN')

if __name__ == "__main__":
    args = parser.parse_args()
    seeds = [args.seed]
    global_results = {}
    for id_r, seed in enumerate(seeds):
        seed_everything(seed, workers=True)

        print("MNIST SVHN - Model {} Run number {} ".format(args.model, id_r))

        modalities_list = [MNIST(latent_dim=latent_dim, reconstruction_weight=r_w_mnist, lhood_name="laplace"),
                           SVHN(latent_dim=latent_dim, reconstruction_weight=r_w_svhn,
                                lhood_name="laplace")
                           ]

        if args.model == "mopoe":

            model_mmvae = MoPoEVAE(
                latent_dim=latent_dim,
                batch_size=batch_size,
                beta=beta,
                modalities_list=modalities_list,
                train_loader=train_loader,
                test_loader=test_loader,
                eval_epoch=eval_epoch,
                do_evaluation=do_evaluation,
                do_fd=do_fd,
                n_fd=5000,
                log_epoch=log_epoch,
                test_batch_size=test_batch_size,
                lr=lr
            )
        elif args.model == "mvae":

            model_mmvae = PoEVAE(model_name=args.model + str(seed),
                                 latent_dim=latent_dim,
                                 batch_size=batch_size,
                                 beta=beta,
                                 modalities_list=modalities_list,
                                 train_loader=train_loader,
                                 test_loader=test_loader,
                                 eval_epoch=eval_epoch,
                                 do_evaluation=do_evaluation,
                                 do_fd=do_fd,
                                 nb_batchs=5,
                                 n_fd=1000,
                                 log_epoch=log_epoch,
                                 test_batch_size=test_batch_size,
                                 lr=lr
                                 )
        elif args.model == "mmvae":

            model_mmvae = MoEVAE(model_name=args.model + str(seed),
                                 latent_dim=latent_dim,
                                 batch_size=batch_size,
                                 beta=beta,
                                 modalities_list=modalities_list,
                                 train_loader=train_loader,
                                 test_loader=test_loader,
                                 eval_epoch=eval_epoch,
                                 do_evaluation=do_evaluation,
                                 do_fd=do_fd,
                                 n_fd=5000,
                                 log_epoch=log_epoch,
                                 test_batch_size=test_batch_size,
                                 lr=lr
                                 )
        elif args.model == "tcmvae":
            model_mmvae = MVTCAE(model_name=args.model + str(seed),
                                  latent_dim=latent_dim,
                                  batch_size=batch_size,
                                  beta=beta,
                                  modalities_list=modalities_list,
                                  train_loader=train_loader,
                                  test_loader=test_loader,
                                  eval_epoch=eval_epoch,
                                  do_evaluation=do_evaluation,
                                  do_fd=do_fd,
                                  n_fd=5000,
                                  log_epoch=log_epoch,
                                  test_batch_size=test_batch_size,
                                  tc_ratio=5/6,
                                  lr=lr
                                  )
        elif args.model == "nexus":
            latent_dim_mnist = 16
            latent_dim_svhn = 64
            modalities_list = [MNIST(latent_dim=latent_dim_mnist, reconstruction_weight=r_w_mnist, lhood_name="laplace"),
                               SVHN(
                latent_dim=latent_dim_svhn, reconstruction_weight=r_w_svhn, lhood_name="laplace")
            ]
            model_mmvae = Nexus_impl(model_name=args.model + str(seed),
                                     latent_dim=20,
                                     batch_size=batch_size,
                                     beta=1,
                                     modalities_list=modalities_list,
                                     train_loader=train_loader,
                                     test_loader=test_loader,
                                     eval_epoch=eval_epoch,
                                     do_evaluation=do_evaluation,
                                     do_fd=do_fd,
                                     n_fd=50,
                                     log_epoch=log_epoch,
                                     test_batch_size=test_batch_size,
                                     dataset="MNISTSVHN",
                                     max_epoch=20,
                                     lr=lr
                                     )
        elif args.model =="mmvaeplus":

            latent_dim = 10
            latent_dim_w= 10
            beta = 2.5
            eval_epoch = 75
            batch_size = 64
            NUM_epoch = 150
            K = 1
            elbo = "iwae"

            r_w_mnist = 3.92
            r_w_svhn = 1.0
            modalities_list =  [MNIST(latent_dim=latent_dim,lhood_name="laplace",deterministic=False,
                                reconstruction_weight = r_w_mnist,
                                      distengled= True,
                                           latent_dim_w = latent_dim_w) ,
                                SVHN(latent_dim=latent_dim,lhood_name="laplace",deterministic=True, 
                                reconstruction_weight = r_w_svhn,
                                     distengled= True,
                                           latent_dim_w = latent_dim_w)]
                
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


        tb_logger = TensorBoardLogger(save_dir=CHECKPOINT_DIR,
                                      name=str(seed),
                                      log_graph=True)

        trainer = pl.Trainer(
            logger=tb_logger,
            accelerator='gpu',
            devices=1,
            max_epochs=NUM_epoch,
            default_root_dir=CHECKPOINT_DIR,
            num_sanity_val_steps=0,
            deterministic=True,
   #          resume_from_checkpoint="/home/bounoua/work/mld/trained_models/September_beta/MNISTSVHN/mmvaeplus_dreg_k_10/{}/version_0/checkpoints/epoch=49-step=219000.ckpt".format(args.seed)
            #    resume_from_checkpoint="trained_models/MNISTSVHN/nexus5seeds/version_3/checkpoints/epoch=56-step=249660.ckpt"
        )

        trainer.fit(model=model_mmvae, train_dataloaders=train_loader,
                    val_dataloaders=test_loader,
                     )

        results = model_mmvae.final_results
        results["logdir"] = model_mmvae.logdir
