"""
File: main.py
Project: project
Created Date: 2023-08-11 03:46:36
Author: chenkaixu
-----
Comment:
This project were based the pytorch, pytorch lightning and pytorch video library,
for rapid development.
The project to predict Lung figure motion trajectory.

Have a good code time!
-----
Last Modified: Friday October 11th 2024 1:04:43 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

10-01-2024	Kaixu Chen clean the code.
2023-11-20 Chen change the tensorboard logger save path.

This is Local version

"""

import os, warnings, logging
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers

# callbacks
from pytorch_lightning.callbacks import (
    TQDMProgressBar,
    RichModelSummary,
    ModelCheckpoint,
    lr_monitor,
)
from pl_bolts.callbacks import PrintTableMetricsCallback, TrainingDataMonitor

# from utils.utils import get_ckpt_path

from project.dataloader.data_loader import CTDataModule
from project.train import PredictLightningModule

import hydra
from omegaconf import DictConfig


# %%
# @hydra.main(version_base=None, config_path="/home/ec2-user/SeqX2Y_PyTorch/configs", config_name="config.yaml")
def train(hparams: DictConfig):
    # set seed
    seed_everything(42, workers=True)

    # load train process
    ConvLSTMmodel = PredictLightningModule(hparams)

    # instance the data module
    data_module = CTDataModule(hparams.train, hparams.data)

    # for the tensorboard
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=hparams.train.log_path, name="tensorboard_logs"
    )

    callbacks = [
        lr_monitor.LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=100),
        RichModelSummary(max_depth=2),
        # define the checkpoint becavier.
        ModelCheckpoint(
            filename="{epoch}-{val_loss:.2f}",
            auto_insert_metric_name=True,
            monitor="val_loss",
            mode="min",
            save_last=True,
            save_top_k=3,
        ),
        # bolts callbacks
        PrintTableMetricsCallback(),
        # monitor = TrainingDataMonitor(log_every_n_steps=1)
    ]

    trainer = Trainer(
        devices=[hparams.train.gpu_num,],
        accelerator="gpu",
        max_epochs=hparams.train.max_epochs,
        logger=tb_logger,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
    )

    trainer.fit(ConvLSTMmodel, data_module)

@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def init_params(config):
    # path to list
    print(config)
    # feed config to train
    train(config)


# %%
if __name__ == "__main__":
    logging.info("Training Start!")
    init_params()
    # train()
    logging.info("Training finish!")
