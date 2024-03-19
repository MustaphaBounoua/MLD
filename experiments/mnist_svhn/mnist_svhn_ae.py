from src.dataLoaders.MnistSvhnText.modalities import MNIST, SVHN, LABEL
from src.utils import create_forlder
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from src.dataLoaders.MnistSvhnText.MnistSvhnText import get_data_set_svhn_mnist
from src.unimodal.BaseAE import AE
import logging
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os


logging.getLogger("lightning").setLevel(logging.ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

mod = "mnist"

parser.add_argument('--modality', type=str, default="mnist",
                    help=("Options possible are mnist, svhn"))

NUM_epoch = 150
batch_size = 128
lr = 0.001


train, test = get_data_set_svhn_mnist(with_text=False)

train_loader = DataLoader(train, batch_size=batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size,
                         shuffle=True,
                         num_workers=8, drop_last=False)


CHECKPOINT_DIR = "trained_models/"
create_forlder(CHECKPOINT_DIR)
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'MNISTSVHN')


if __name__ == "__main__":

    dim_mnist = 16
    dim_svhn = 64

    args = parser.parse_args()

    if args.modality == "mnist":
        modality = MNIST(latent_dim=dim_mnist,
                         deterministic=True, lhood_name="laplace")

    else:
        modality = SVHN(latent_dim=dim_svhn,  deterministic=True,
                        lhood_name="laplace")

    model = AE(modality=modality,
               regularization=None, train_samp = next(iter(train_loader))[0][args.modality],
               test_samp=next(iter(test_loader))[0][args.modality])

    print(model)

    tb_logger = TensorBoardLogger(save_dir=CHECKPOINT_DIR,
                                  name=str(
                                      "ae"+args.modality)
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
