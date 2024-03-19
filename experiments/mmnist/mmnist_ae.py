

from src.unimodal.BaseAE import AE
from src.dataLoaders.MMNIST.MMNIST import get_mmnist_dataset
from src.dataLoaders.MMNIST.modalities import MMNIST
from src.utils import create_forlder
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
import sys
root = "/home/bounoua/work/mld/"
sys.path.append(root)


logging.getLogger("lightning").setLevel(logging.ERROR)


logging.getLogger("lightning").setLevel(logging.ERROR)


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)


parser.add_argument('--modality', type=str, default="m0",
                    help=("Options possible are image, sound, trajectory"))


NUM_epoch = 300
batch_size = 128
beta = 1.0
lr = 0.001

train, test = get_mmnist_dataset()
train_loader = DataLoader(train, batch_size=batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True)

test_loader = DataLoader(test, batch_size=512,
                         shuffle=True,
                         num_workers=4, drop_last=True)


CHECKPOINT_DIR = "trained_models/"
create_forlder(CHECKPOINT_DIR)
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'MMNIST')


if __name__ == "__main__":

    args = parser.parse_args()
    latent_dim = 160
    modalities_list = [MMNIST(latent_dim=latent_dim, lhood_name="laplace", resnet=False,
                              deterministic=True, name="m{}".format(i)) for i in [0, 1, 2, 3, 4]]

    for modality in modalities_list:
        if modality.name == args.modality:
            model = AE(modality=modality,
                      train_samp = next(iter(train_loader))[0][modality.name],
                        test_samp=next(iter(test_loader))[0][modality.name],
                       lr=lr,
                       regularization=None)

            tb_logger = TensorBoardLogger(save_dir=CHECKPOINT_DIR,
                                          name=str(
                                              "ae_deter_latent"+modality.name)
                                          )

            trainer = pl.Trainer(
                logger=tb_logger,
                check_val_every_n_epoch=1,
                accelerator='gpu',
                devices=1,
                max_epochs=NUM_epoch,
                default_root_dir=CHECKPOINT_DIR
            )
            trainer.fit(model=model, train_dataloaders=train_loader,
                        val_dataloaders=test_loader)
