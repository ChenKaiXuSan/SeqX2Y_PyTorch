'''
File: train.py
Project: project
Created Date: 2023-08-11 08:48:00
Author: chenkaixu
-----
Comment:
The train and val process for main file.
This file under the pytorch lightning and inherit the lightningmodule.
 
Have a good code time!
-----
Last Modified: Tuesday June 11th 2024 3:08:47 am
Modified By: the developer formerly known as Hao Ouyang at <ouyanghaomail@gmail.com>
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

10-01-2024	Kaixu Chen	add the 3D CNN to process the time series 3D image.
2023-09-26	KX.C	change the train and val process, here we think need use the self.seq to control the seq_len, to reduce the memory usage.

'''

# %%
import os, csv, logging, shutil
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchmetrics
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from pytorch_lightning import LightningModule
from torchmetrics import classification

# from models.seq2seq_4DCT_voxelmorph import EncoderDecoderConvLSTM
# from models.lite_seq2seq_4DCT_voxelmorph import EncoderDecoderConvLSTM
from models.Time_series_seq2seq_4DCT_voxelmorph import EncoderDecoderConvLSTM
from models.Warp import Warp
from utils.image_saver import save_dvf_image, save_bat_pred_image, save_sitk_images, save_sitk_DVF_images
from loss_analyst import *

# %%
class PredictLightningModule(LightningModule):

    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.img_size = hparams.data.img_size # from data/4DCT.yaml
        self.lr = hparams.optimizer.lr        # from optimizer/adam.yaml
        self.seq = hparams.train.seq          # from config.yaml
        self.vol = hparams.train.vol          # from config.yaml

        self.model = EncoderDecoderConvLSTM(
            # nf=96, in_chan=1, size1=30, size2=176, size3=140)
            #  nf=96, in_chan=1, size1=70, size2=120, size3=140)
            # ! FIXME  # input4 output4 93 max
            nf = 96, in_chan=1, size1=self.vol, size2=self.img_size, size3=self.img_size) 
            # ! FIXME 为什么108和120消耗的GPU一样？但是在108到120之间的消耗又少？
            # seq=3,98:21.58, 100:21.95, 102:22.43, 104:22.73, 106:23.17, 108:23.57, 109:22.43, 110:22.57, 112:22.91, 114:23.35, 116:22.83, 118:23.36, 120:23.58
            # nf=96, in_chan=1, size1=30, size2=256, size3=256)

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        # select the metrics
        self._accuracy = classification.MulticlassAccuracy(num_classes=4)
        self._precision = classification.MulticlassPrecision(num_classes=4)
        self._confusion_matrix = classification.MulticlassConfusionMatrix(num_classes=4)
        # 在您的模型初始化中 select the metrics
        self.mse = torchmetrics.MeanSquaredError()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.r2_score = torchmetrics.R2Score()

        # to save the True First time train loss and val loss
        self.initial_train_loss_set = False  # lightning框架会进行一次检查，会有值产生，但是不能使用这个值，所以用一个标志来跟踪是否已经完成了第一次实际训练迭代
        self.initial_val_loss_set = False  # 同上，标志，表示是否设置了初始验证损失
        self.initial_train_loss = None # train loss
        self.initial_val_loss = None # val loss

    def forward(self, x):
        return self.model(x)
    

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        '''
        main.py中的trainer.fit(ConvLSTMmodel, data_module)会自动匹配网络模型(即ConvLSTMmodel实例化后的PredictLightningModule)和dataset(即data_module实例化后的CTDataModule),
        匹配后, 这里的training_step会和data_loader.py中 CTDataModule 的 train_dataloader 相匹配, batch会接收来自train_dataloader return来的[DataLoader1, DataLoader2],
        同理, validation_step会和 val_dataloader相匹配...

        train steop when trainer.fit called

        sample_info_dict = {
            'patient_id': idx,
            '4DCT': torch.stack(one_patient_full_vol, dim=0),
            '2D_time_series': torch.stack(one_patient_time_series, dim=0) # seq, c, h, w
        }

        Args:
            batch (torch.Tensor): is sample info dict from dataloader.
            batch_idx (int): batch index, or patient index
        Returns: train loss, pytorch lightning will auto backward and update the model.
        
        '''

        # unpack the batch
        ct_data = batch['4DCT']
        time_series_img = batch['2D_time_series']

        b, seq, c, vol, h, w = ct_data.size()
        b, c, t, h, w = time_series_img.size()

        invol = ct_data.clone().detach()

        # pred the video frames
        # invol: 1, 1, 1, 128, 128, 128 # b, seq, c, vol, h, w
        # time_series_img: 1, 1, 3, 128, 128 # b, c, seq (f), h, w
        bat_pred, DVF = self.model(invol, time_series_img, future_seq=self.seq)  

        # Caluate training loss
        train_loss = calculate_train_loss(bat_pred, DVF, ct_data, seq)
        
        # Storing train loss on the True first iteration 确保只在第一次实际训练迭代时设置初始训练损失
        if not self.initial_train_loss_set:
            self.initial_train_loss = train_loss.detach().clone()
            self.initial_train_loss_set = True
        relative_train_loss = train_loss / self.initial_train_loss
        #save logs
        logging.info("Patient index: %s" % (batch_idx))
        self.log('train_loss', relative_train_loss, on_epoch=True, on_step=True)
        logging.info('train_loss: %.4f' % relative_train_loss)
        print("Current train_loss:", train_loss.item())
        #!FIXME Metrics Test But Erro ValueError: Expected both prediction and target to be 1D or 2D tensors, but received tensors with dimension torch.Size([1, 1, 118, 128, 128]) 
        # self.log('train_mse', mse_value, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('train_mae', mae_value, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('train_r2', r2_value, on_step=True, on_epoch=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        '''
        val step when trainer.fit called.

        sample_info_dict = {
            'patient_id': idx,
            '4DCT': torch.stack(one_patient_full_vol, dim=0),
            '2D_time_series': torch.stack(one_patient_time_series, dim=0) # seq, c, h, w
        }

        Args:
            batch (torch.Tensor): is sample info dict from dataloader.
            batch_idx (int): batch index, or patient index
        Returns: None
        '''

        # unpack the batch
        ct_data = batch['4DCT']
        save_sitk_images(ct_data, batch_idx, '/workspace/SeqX2Y_PyTorch/test/Imageresult/GT') # Save the croped GT images
        time_series_img = batch['2D_time_series']

        b, seq, c, vol, h, w = ct_data.size()
        b, c, t, h, w = time_series_img.size()

        invol = ct_data.clone().detach()

        # pred the video frames
        with torch.no_grad():
            # invol: 1, 4, 1, 128, 128, 128 # b, seq, c, vol, h, w
            # time_series_img: 1, 4, 3, 128, 128 # b, seq (f), c, h, w
            # bat_pred, DVF = self.model(invol, rpm_x=test_x_rpm_tensor, rpm_y=test_y_rpm_tensor, future_seq=self.seq)  # [1,2,3,176,176]
            bat_pred, DVF = self.model(invol, time_series_img, future_seq=self.seq)  # [1,2,3,176,176]
            # bat_pred.shape=(1,1,3,128,128,128) DVF.shape=(1,3,3,128,128,128) 

        # Save images
        # save_dvf_image(DVF, batch_idx, '/workspace/SeqX2Y_PyTorch/test/Imageresult')
        # save_bat_pred_image(bat_pred, batch_idx, '/workspace/SeqX2Y_PyTorch/test/Imageresult')
        save_sitk_images(bat_pred, batch_idx, '/workspace/SeqX2Y_PyTorch/test/Imageresult')
        save_sitk_DVF_images(DVF, batch_idx, '/workspace/SeqX2Y_PyTorch/test/Imageresult' )


        # calculate the validation loss
        val_loss, ssim_values, ncc_values, dice_values, mae_values = calculate_val_loss(bat_pred, DVF, ct_data, seq)


        # Storing val_loss on the True first iteration 确保只在第一次实际验证迭代时设置初始验证损失
        if not self.initial_val_loss_set:
            self.initial_val_loss = val_loss.detach().clone()
            self.initial_val_loss_set = True          
        relative_val_loss = val_loss / self.initial_val_loss 
        # 
        average_ssim = sum(ssim_values) / len(ssim_values)
        # average_ncc = sum(ncc_values) / len(ncc_values)
        # average_dice = sum(dice_values) / len(dice_values)
        average_mae = sum(mae_values) / len(mae_values) # MAE不取平均值,范围为[0, +∞) 
        # save logs  
        logging.info("Patient index: %s" % (batch_idx)) 
        self.log('val_loss', relative_val_loss, on_epoch=True, on_step=True)
        logging.info('val_loss: %.4f' % relative_val_loss)
        print("Current val_loss:", val_loss.item())
        # print(f"Average SSIM: {average_ssim}")
        self.log('Average SSIM', average_ssim)
        logging.info('Average SSIM: %.4f' % average_ssim)
        # self.log('Average NCC', average_ncc)
        # logging.info('Average NCC: %.4f' % average_ncc)
        # self.log('Average Dice', average_dice)
        # logging.info('Average Dice: %.4f' % average_dice)
        # logging.info('Average Dice: %.4f' % average_dice.item())
        self.log('Average MAE', average_mae)
        logging.info('Average MAE: %.4f' % average_mae)
        # Log each MAE value separately
        # for i, mae in enumerate(mae_values):
        #     self.log(f'MAE Value {i}', mae)
        #     logging.info('MAE Value %d: %.4f' % (i, mae))
        #Draw image
        # draw_image(average_ssim, average_ncc, average_dice, average_mae)


    def configure_optimizers(self):
        '''
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        '''

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                "monitor": "val_loss",
            },
        }

    def _get_name(self):
        return self.model