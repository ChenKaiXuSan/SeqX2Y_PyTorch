"""
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
Last Modified: Friday October 11th 2024 4:31:23 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

10-01-2024	Kaixu Chen	add the 3D CNN to process the time series 3D image.
2023-09-26	KX.C	change the train and val process, here we think need use the self.seq to control the seq_len, to reduce the memory usage.

"""

# %%
import os, csv, logging, shutil
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchmetrics
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from pytorch_lightning import LightningModule
from torchmetrics import classification

# *-----------新增 Pyotrch-Gard-Cam------------*
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from project.utils.grad_cam.grad_cam_2D import GradCAM_2D
# *-----------新增 Pyotrch-Gard-Cam------------*

# from models.seq2seq_4DCT_voxelmorph import EncoderDecoderConvLSTM
# from models.lite_seq2seq_4DCT_voxelmorph import EncoderDecoderConvLSTM
from project.models.Time_series_seq2seq_4DCT_voxelmorph import EncoderDecoderConvLSTM

# from project.models.Time_series_seq2seq_4DCT_GradCam import EncoderDecoderConvLSTM
from project.models.Warp import Warp
from project.utils.image_saver import (
    save_dvf_image,
    save_bat_pred_image,
    save_sitk_images,
    save_sitk_DVF_images,
)
from project.loss_analyst import calculate_train_loss, calculate_val_loss


# %%
class PredictLightningModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.img_size = hparams.data.img_size  # from data/4DCT.yaml
        self.lr = hparams.optimizer.lr  # from optimizer/adam.yaml
        self.seq = hparams.train.seq  # from config.yaml
        self.vol = hparams.train.vol  # from config.yaml

        self.model = EncoderDecoderConvLSTM(
            # nf=96, in_chan=1, size1=30, size2=176, size3=140)
            #  nf=96, in_chan=1, size1=70, size2=120, size3=140)
            # ! FIXME  # input4 output4 93 max
            nf=96,
            in_chan=1,
            size1=self.vol,
            size2=self.img_size,
            size3=self.img_size,
        )
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
        self.initial_train_loss = None  # train loss
        self.initial_val_loss = None  # val loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """
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

        """

        # unpack the batch
        ct_data = batch["4DCT"]
        time_series_img = batch["2D_time_series"]
        # time_series_list = batch['1D_time_series']
        # time_series_list = [[int(item) for item in sublist] for sublist in time_series_list]  # TODO: 将列表中的字符串转换为整数
        # RPM_tensor = torch.Tensor(time_series_list)
        # RPM_tensor = RPM_tensor.cuda()

        b, seq, c, vol, h, w = ct_data.size()
        b, c, t, h, w = time_series_img.size()

        invol = ct_data.clone().detach()

        # pred the video frames
        # invol: 1, 1, 1, 128, 128, 128 # b, seq, c, vol, h, w
        # time_series_img: 1, 1, 3, 128, 128 # b, c, seq (f), h, w
        bat_pred, DVF, _ = self.model(
            invol, time_series_img, future_seq=self.seq
        )  # 原本 bat_pred, DVF = self.model(invol, time_series_img, future_seq=self.seq)

        # Caluate training loss
        train_loss = calculate_train_loss(bat_pred, DVF, ct_data, seq)

        # Storing train loss on the True first iteration 确保只在第一次实际训练迭代时设置初始训练损失
        if not self.initial_train_loss_set:
            self.initial_train_loss = train_loss.detach().clone()
            self.initial_train_loss_set = True
        relative_train_loss = train_loss / self.initial_train_loss
        # save logs
        logging.info("Patient index: %s" % (batch_idx))
        self.log("train_loss", relative_train_loss, on_epoch=True, on_step=True)
        logging.info("train_loss: %.4f" % relative_train_loss)
        print("Current train_loss:", train_loss.item())
        #!FIXME Metrics Test But Erro ValueError: Expected both prediction and target to be 1D or 2D tensors, but received tensors with dimension torch.Size([1, 1, 118, 128, 128])
        # self.log('train_mse', mse_value, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('train_mae', mae_value, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('train_r2', r2_value, on_step=True, on_epoch=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """
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
        """

        # unpack the batch
        ct_data = batch["4DCT"]
        # save_sitk_images(ct_data, batch_idx, '/workspace/SeqX2Y_PyTorch/test/Imageresult/GT') # Save the croped GT images
        # save_sitk_images(ct_data, batch_idx, '/home/ec2-user/SeqX2Y_PyTorch/test/Imageresult/GT') 移到下面去了
        time_series_img = batch["2D_time_series"]
        # time_series_list = batch['1D_time_series'] # TODO: 这里改好了，你只需要把这个坐标传到model里面就行了
        # time_series_list = [[int(item) for item in sublist] for sublist in time_series_list]  # TODO: 将列表中的字符串转换为整数
        # RPM_tensor = torch.Tensor(time_series_list)
        # RPM_tensor = RPM_tensor.cuda()

        b, seq, c, vol, h, w = ct_data.size()
        b, c, t, h, w = time_series_img.size()

        invol = ct_data.clone().detach()

        # *-----------新增 Pyotrch-Gard-Cam------------*
        # 选择要应用Grad-CAM的目标卷积层，比如conv4层
        target_layers = [self.model.encoder3d_cnn.conv4]

        # 初始化Grad-CAM，指定模型和目标层
        cam = GradCAM_2D(model=self.model, target_layers=target_layers)
        # *-----------新增 Pyotrch-Gard-Cam------------*

        # pred the video frames
        with torch.no_grad():
            # invol: 1, 4, 1, 128, 128, 128 # b, seq, c, vol, h, w
            # time_series_img: 1, 4, 3, 128, 128 # b, seq (f), c, h, w
            # bat_pred, DVF = self.model(invol, rpm_x=test_x_rpm_tensor, rpm_y=test_y_rpm_tensor, future_seq=self.seq)  # [1,2,3,176,176]
            # bat_pred, DVF = self.model(invol, time_series_img, future_seq=self.seq)  # [1,2,3,176,176] 没加pytorch-gard-cam的时候
            (
                bat_pred,
                DVF,
                time_series_fat_encoder,
                time_series_fat_decoder,
                batch_2D,
            ) = self.model(
                invol, time_series_img, future_seq=self.seq
            )  # 加pytorch-gard-cam
            # bat_pred.shape=(1,1,3,128,128,128) DVF.shape=(1,3,3,128,128,128)

            # *-----------新增 Pyotrch-Gard-Cam------------*
            # 计算grad cam的输入需要和目标层的输入保持一致，这里需要根据你的实际输入数据调整
            input_tensor = {
                "invol": invol,
                "time_series_img": time_series_img,
                "future_seq": self.seq,
            }

            # 选择目标类（对于分类任务，一般是类别索引）, 这里需要根据你的任务修改，假设使用类索引0
            # targets = [ClassifierOutputTarget(0)]  # 示例：类别索引为0
            targets = {"bat_pred": bat_pred, "DVF": DVF}

            # 计算目标层的Grad-CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # 将Grad-CAM灰度图映射到原图像上以进行可视化, 你需要根据实际的输入图像进行映射
            visualization = show_cam_on_image(
                time_series_fat_encoder, grayscale_cam, use_rgb=True
            )

            # 可视化或保存CAM结果
            plt.imshow(visualization)
            plt.show()
            plt.savefig(
                "/home/ec2-user/SeqX2Y_PyTorch/test/Imageresult/grad_cam/visualization.png"
            )
            # *-----------新增 Pyotrch-Gard-Cam------------*

        # Save images
        # save_dvf_image(DVF, batch_idx, '/workspace/SeqX2Y_PyTorch/test/Imageresult')
        # save_bat_pred_image(bat_pred, batch_idx, '/workspace/SeqX2Y_PyTorch/test/Imageresult')
        save_sitk_images(
            ct_data, batch_idx, "/home/ec2-user/SeqX2Y_PyTorch/test/Imageresult/GT"
        )  # save GT img 1
        save_sitk_images(
            bat_pred, batch_idx, "/home/ec2-user/SeqX2Y_PyTorch/test/Imageresult"
        )  # save pred img 2
        save_sitk_DVF_images(
            DVF, batch_idx, "/home/ec2-user/SeqX2Y_PyTorch/test/Imageresult"
        )  # save DVF img 3

        # calculate the validation loss
        val_loss, ssim_values, ncc_values, dice_values, mae_values, psnr_values = (
            calculate_val_loss(bat_pred, DVF, ct_data, seq)
        )

        # Storing val_loss on the True first iteration 确保只在第一次实际验证迭代时设置初始验证损失
        if not self.initial_val_loss_set:
            self.initial_val_loss = val_loss.detach().clone()
            self.initial_val_loss_set = True
        relative_val_loss = val_loss / self.initial_val_loss
        #
        average_ssim = sum(ssim_values) / len(ssim_values)
        # average_ncc = sum(ncc_values) / len(ncc_values)
        # average_dice = sum(dice_values) / len(dice_values)
        average_mae = sum(mae_values) / len(mae_values)  # MAE不取平均值,范围为[0, +∞)
        average_psnr = sum(psnr_values) / len(psnr_values)
        # save logs
        logging.info("Patient index: %s" % (batch_idx))
        self.log("val_loss", relative_val_loss, on_epoch=True, on_step=True)
        logging.info("val_loss: %.4f" % relative_val_loss)
        print("Current val_loss:", val_loss.item())
        # print(f"Average SSIM: {average_ssim}")
        self.log("Average SSIM", average_ssim)
        logging.info("Average SSIM: %.4f" % average_ssim)
        # self.log('Average NCC', average_ncc)
        # logging.info('Average NCC: %.4f' % average_ncc)
        # self.log('Average Dice', average_dice)
        # logging.info('Average Dice: %.4f' % average_dice)
        # logging.info('Average Dice: %.4f' % average_dice.item())
        self.log("Average MAE", average_mae)
        logging.info("Average MAE: %.4f" % average_mae)
        # Log each MAE value separately
        # for i, mae in enumerate(mae_values):
        #     self.log(f'MAE Value {i}', mae)
        #     logging.info('MAE Value %d: %.4f' % (i, mae))
        self.log("Average PSNR", average_psnr)
        logging.info("Average PSNR: %.4f" % average_psnr)
        # Draw image
        # draw_image(average_ssim, average_ncc, average_dice)

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        """
        考虑在这里进行测试步骤，从val step中分割出来

        Args:
            batch (torch.Tensor): batch data
            batch_idx (int): batch index
        """

        pass

    def configure_optimizers(self):
        """
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        """

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
