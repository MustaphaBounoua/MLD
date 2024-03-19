

from src.dataLoaders.MnistSvhnText.modalities import MNIST, SVHN, LABEL
from src.models.LateFusionAE import LateFusionAE
from src.models.MLD import MLD
from src.unimodal.BaseAE import AE
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from src.dataLoaders.MnistSvhnText.MnistSvhnText import get_data_set_svhn_mnist
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)


parser.add_argument('--model', type=str, default="mmld",
                    help=("Options possible are mvae, mmvae, mopoe"))

parser.add_argument('--d', type=str, default=0.5,
                    help=("Options possible are "))

parser.add_argument('--seed', type=str, default=1,
                    help=("Options possible are "))

do_evaluation = True
do_fd = True
do_class = False
eval_epoch = 1
log_epoch = 20


NUM_epoch = 151
batch_size = 128

r_w_mnist = 1.0
r_w_svhn = 1.0
r_w_text = 1.0
lr = 1e-4

PATHS = {
    "mnist": "trained_models/MNISTSVHN/aemnist/version_2/checkpoints/epoch=3-step=35040.ckpt",
    "svhn": "trained_models/MNISTSVHN/aesvhn/version_0/checkpoints/epoch=10-step=96360.ckpt"}


train, test = get_data_set_svhn_mnist(with_text=False)

train_loader = DataLoader(train, batch_size=batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True, 
                          pin_memory=True)

test_loader = DataLoader(test, batch_size=512,
                         shuffle=True,
                         num_workers=8, drop_last=True, 
                         pin_memory=True)


CHECKPOINT_DIR = "trained_models/"

CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'MNISTSVHN')


if __name__ == "__main__":

    args = parser.parse_args()

    dim_mnist = 16
    dim_svhn = 64
    dim_label = 10

    modalities_list = [
        MNIST(latent_dim=dim_mnist, reconstruction_weight=r_w_mnist,
              deterministic=True, lhood_name="laplace"),
        SVHN(latent_dim=dim_svhn, reconstruction_weight=r_w_svhn,
             deterministic=True, lhood_name="laplace")
    ]
    aes = []
    for mod in modalities_list:
        aes.append(
            AE.load_from_checkpoint(PATHS[mod.name], modality=mod)
        )

    aes_model = LateFusionAE(aes=aes)
    aes_model.eval()

    d = args.d

    seeds = [int(args.seed)]

    for seed in seeds:

        seed_everything(seed)

        model = MLD(
            aes=aes_model,
            model_name="mld",
            batch_size=batch_size,
            train_loader=train_loader,
            test_loader=test_loader,
            eval_epoch=eval_epoch,
            do_evaluation=do_evaluation,
            do_fd=do_fd,
            nb_samples=8,
            n_fd=5000,
            log_epoch=log_epoch,
            lr=lr,
            nb_batchs=2,
            init_dim=256,
            do_class=False,
            time_dim=256,
            unet_architecture=(1, 1),
            unet_type='linear',
            d=float(d),
            preprocess_type="modality",
            preprocess_op="standerdize",
            check_stat=False,
            group_norm=8,
            betas=[0.1, 20],
            train_batch_size=512,
            N_step=250,
            train_latent=False,
            importance_sampling=False,
            ll_weight=False,
            debug=False,
            use_attention=False,
            shift_scale=False,
            num_head=3,
            use_ema=True,
            cross_gen="repaint"
        )
        CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'MLD')

        tb_logger = TensorBoardLogger(save_dir=os.path.join(CHECKPOINT_DIR, "d" + str(args.d)),
                                      name=str(seed))

     

        # print(model)
        trainer = pl.Trainer(
            logger=tb_logger,
            check_val_every_n_epoch=5,
            accelerator='gpu',
            devices=1,
            max_epochs=NUM_epoch,
            default_root_dir=CHECKPOINT_DIR,
            num_sanity_val_steps=0,
            deterministic=True,


            #     strategy="ddp",
        )

        trainer.fit(model=model, train_dataloaders=model.train_loader,
                    val_dataloaders=model.test_loader
                    )
