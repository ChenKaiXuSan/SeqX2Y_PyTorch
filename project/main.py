'''
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
Last Modified: Tuesday June 11th 2024 3:08:47 am
Modified By: the developer formerly known as Hao Ouyang at <ouyanghaomail@gmail.com>
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

10-01-2024	Kaixu Chen clean the code.
2023-11-20 Chen change the tensorboard logger save path.

This is Local version

'''

# %%
import os, warnings, logging
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers

# callbacks
from pytorch_lightning.callbacks import TQDMProgressBar, RichModelSummary, ModelCheckpoint, lr_monitor
from pl_bolts.callbacks import PrintTableMetricsCallback, TrainingDataMonitor
# from utils.utils import get_ckpt_path

from dataloader.data_loader import CTDataModule
from train import PredictLightningModule

import hydra
from omegaconf import DictConfig

# %%
# @hydra.main(version_base=None, config_path="/workspace/SeqX2Y_PyTorch/configs", config_name="config.yaml")
def train(hparams: DictConfig):

    # set seed
    seed_everything(42, workers=True)

    # load train process
    ConvLSTMmodel = PredictLightningModule(hparams)

    # instance the data module
    data_module = CTDataModule(hparams.train, hparams.data)

    # for the tensorboard
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=hparams.train.log_path, 
                                            name= "tensorboard_logs")

    lr_logger = lr_monitor.LearningRateMonitor(logging_interval='step')

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

    # bolts callbacks
    table_metrics_callback = PrintTableMetricsCallback()
    monitor = TrainingDataMonitor(log_every_n_steps=1)

    trainer = Trainer(
        devices=[hparams.train.gpu_num,],
        accelerator="gpu",
        max_epochs=hparams.train.max_epochs,
        logger=tb_logger,
        check_val_every_n_epoch=1,
        callbacks=[progress_bar, rich_model_summary, table_metrics_callback,
                   model_check_point, lr_logger], # 去掉monitor
    )

    trainer.fit(ConvLSTMmodel, data_module)

@hydra.main(version_base=None, config_path="/workspace/SeqX2Y_PyTorch/configs", config_name="config.yaml")
def init_params(config):
    # path to list 
    print(config)
    # feed config to train
    train(config)

# %%
if __name__ == '__main__':

    logging.info("Training Start!")
    init_params()
    # train()
    logging.info("Training finish!")
