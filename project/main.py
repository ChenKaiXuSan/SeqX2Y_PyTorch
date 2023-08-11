'''
this project were based the pytorch, pytorch lightning and pytorch video library, 
for rapid development.
'''

# %%
import os
import warnings
import logging
import pytorch_lightning
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
# callbacks
from pytorch_lightning.callbacks import TQDMProgressBar, RichModelSummary, RichProgressBar, ModelCheckpoint, EarlyStopping
from pl_bolts.callbacks import PrintTableMetricsCallback, TrainingDataMonitor
# from utils.utils import get_ckpt_path

from dataloader.data_loader import CTDataModule
from train import PredictLightningModule

from argparse import ArgumentParser
import hydra
from omegaconf import DictConfig

# %%


@hydra.main(version_base=None, config_path="/workspace/SeqX2Y_PyTorch/configs", config_name="config.yaml")
def train(hparams: DictConfig):

    # set seed
    seed_everything(42, workers=True)

    # load train process
    ConvLSTMmodel = PredictLightningModule(hparams)

    # instance the data module
    data_module = CTDataModule(hparams.train, hparams.data)

    # for the tensorboard
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(
        hparams.train.log_path), name=hparams.train.version)

    # some callbacks
    progress_bar = TQDMProgressBar(refresh_rate=100)
    rich_model_summary = RichModelSummary(max_depth=2)

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        filename='{epoch}-{val_loss:.2f}',
        auto_insert_metric_name=True,
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=3,
    )

    # define the early stop.
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
    )

    # bolts callbacks
    table_metrics_callback = PrintTableMetricsCallback()
    monitor = TrainingDataMonitor(log_every_n_steps=50)

    trainer = Trainer(
        devices=[hparams.train.gpu_num,],
        accelerator="gpu",
        max_epochs=hparams.train.max_epochs,
        logger=tb_logger,
        check_val_every_n_epoch=1,
        callbacks=[progress_bar, rich_model_summary, table_metrics_callback,
                   monitor, model_check_point,],
    )

    trainer.fit(ConvLSTMmodel, data_module)

    # Acc_list = trainer.validate(classification_module, data_module, ckpt_path='best')

    # return the best acc score.
    # return model_check_point.best_model_score.item()


# %%
if __name__ == '__main__':

    logging.info("Training Start!")
    train()
    logging.info("Training finish!")
