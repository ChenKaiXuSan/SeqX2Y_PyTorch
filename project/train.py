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
Last Modified: Tuesday April 23rd 2024 2:16:03 am
Modified By: the developer formerly known as Hao Ouyang at <ouyanghaomail@gmail.com>
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------
2023-09-26	KX.C	change the train and val process, here we think need use the self.seq to control the seq_len, to reduce the memory usage.

'''

# %%
import os, csv, logging, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torchmetrics
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from pytorch_lightning import LightningModule
from torchmetrics import classification

from models.seq2seq_4DCT_voxelmorph import EncoderDecoderConvLSTM
# from models.lite_seq2seq_4DCT_voxelmorph import EncoderDecoderConvLSTM
from models.Warp import Warp
from utils.image_saver import save_dvf_image, save_bat_pred_image, save_sitk_images, save_sitk_DVF_images

# %%
# Caluate psnr values
def psnr(target, ref):
    # Calculate MSE using torch.nn.functional
    mse = F.mse_loss(target, ref, reduction='mean')
    if mse == 0:
        return torch.tensor(float('inf'))
    dynamic_range = 4000  # 从-1024到2976的范围
    return 20 * torch.log10(dynamic_range / torch.sqrt(mse))

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
            # ! FIXME
            nf = 96, in_chan=1, size1=self.vol, size2=self.img_size, size3=self.img_size) #88
            # nf=96, in_chan=1, size1=30, size2=256, size3=256)

        # TODO you should generate rpm.csv file by yourself.
        # load RPM
        with open(hparams.test['rpm'], 'r') as f:
            self.data = list(csv.reader(f, delimiter=","))

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
    
    # Calculate smoothness loss for DVF
    def calculate_smoothness_loss(self,dvf):
        # Assuming dvf is a 6D tensor: batch x seq x channels x depth x height x width
        dvf_grad_x = torch.gradient(dvf, dim=4, spacing=(1,))[0]
        dvf_grad_y = torch.gradient(dvf, dim=3, spacing=(1,))[0]
        dvf_grad_z = torch.gradient(dvf, dim=2, spacing=(1,))[0]
        # Summing the squares of the gradients
        smoothness_loss = dvf_grad_x.pow(2) + dvf_grad_y.pow(2) + dvf_grad_z.pow(2)
        # smoothness_loss = dvf_grad_x**2 + dvf_grad_y**2 + dvf_grad_z**2
        # Taking the mean over all dimensions except the batch
        return smoothness_loss.mean(dim=[1, 2, 3, 4])
    
    # def calculate_ssim(self, x, y):
    #     _, _, depth, _, _ = x.shape
    #     ssim_scores = []

    #     # 确保x和y转换为numpy数组，因为skimage的SSIM函数需要numpy数组
    #     x_np = x[0].cpu().detach().numpy()
    #     y_np = y[0].cpu().detach().numpy()

    #     # 遍历每个深度切片
    #     for d in range(depth):
    #         # 计算每个深度切片的SSIM
    #         ssim_value = SSIM(x_np[:, d, ...], y_np[:, d, ...])
    #         ssim_scores.append(ssim_value)

    #     # 返回所有深度切片的平均SSIM
    #     return sum(ssim_scores) / len(ssim_scores)
    
    # # Calculate NCC values: All dimensions together
    # def normalized_cross_correlation(self, x, y):
    #     mean_x = torch.mean(x) # x torch.Size([1, 1, 118, 128, 128])
    #     mean_y = torch.mean(y) # y torch.Size([1, 1, 118, 128, 128])
    #     x_normalized = x - mean_x
    #     y_normalized = y - mean_y
    #     ncc = torch.sum(x_normalized * y_normalized) / (torch.sqrt(torch.sum(x_normalized ** 2)) * torch.sqrt(torch.sum(y_normalized ** 2)))
    #     return ncc

    # Calculate NCC values: Depth dimension only
    def normalized_cross_correlation(self, x, y):
        # batch_size = x.shape[2]
        batch_size, channels, depth, _, _ = x.shape
        ncc_scores = []

        # Traverse each sample in depth
        for d in range(depth):
            # Average
            mean_x = torch.mean(x[:,:,d,...])
            mean_y = torch.mean(y[:,:,d,...])
            # normalized
            x_normalized = x[:,:,d,...] - mean_x
            y_normalized = y[:,:,d,...] - mean_y
            # NCC
            ncc = torch.sum(x_normalized * y_normalized) / (torch.sqrt(torch.sum(x_normalized ** 2)) * torch.sqrt(torch.sum(y_normalized ** 2)))
            ncc_scores.append(ncc.item())
        # Average NCC
        return sum(ncc_scores) / len(ncc_scores)

    # # Calculate Dice values: All dimensions together
    # def dice_coefficient(self, pred, target):
    #     smooth = 1.0  # Used to prevent division by zero
    #     # Binarize the prediction and target, and the threshold is usually set at 0.5
    #     pred = (pred > 0).float()
    #     target = (target > 0).float()
    #     # intersection = (pred * target).sum()
    #     intersection = torch.sum(pred*target, dim=[2, 3, 4])
    #     # dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    #     dice = (2. * intersection + smooth) / (torch.sum(pred, dim=[2, 3, 4]) + torch.sum(target, dim=[2, 3, 4]) + smooth)
    #     return dice
    
    # Calculate Dice values: Depth only
    def dice_coefficient(self, x, y):
        _, _, depth, _, _ = x.shape
        dice_scores = []

        # Convert x and y values from [-1, 1] to [0, 1]
        x = (x > 0).float()
        y = (y > 0).float()

        # Traverse each sample in depth
        for d in range(depth):
            # intersection
            intersection = (x[:, :, d, ...] * y[:, :, d, ...]).sum()
            # Dice
            dice = (2. * intersection) / (x[:, :, d, ...].sum() + y[:, :, d, ...].sum())
            dice_scores.append(dice.item())
        # Average Dice
        return sum(dice_scores) / len(dice_scores)
    
    # Calculate TRE values
    def calculate_tre(self, points_pred, points_true):
        """
        :param points_pred: 预测的点的位置，形状为 [N, 3]，其中 N 是点的数量
        :param points_true: 真实的点的位置，形状为 [N, 3]
        :return: TRE 值
        """
        # 确保预测点和真实点的数量相同
        assert points_pred.shape == points_true.shape, "预测点和真实点的数量和/或维度不匹配"
        # 计算点对之间的欧氏距离
        tre = torch.sqrt(torch.sum((points_pred - points_true) ** 2, dim=1))
        return torch.mean(tre)

    def training_step(self, batch: torch.Tensor, batch_idx:int):
        '''
        train steop when trainer.fit called

        Args:
            batch (torch.Tensor): b, seq, vol, c, h, w
            batch_idx (int):batch index.

        Returns: None
        '''

#------------------------- New ---------------------------------
        b, seq, c, vol, h, w = batch.size()
        # Reading RPM #
        with open('/home/ec2-user/SeqX2Y_PyTorch/dataset/Diagram_Coordinates/1D_rpm.csv', 'r', encoding='utf-8-sig') as f:  # 使用'utf-8-sig'来自动处理BOM
            data = list(csv.reader(f, delimiter=","))        
        RPM = np.array(data)
        RPM = np.float32(RPM)

        select_seq = RPM[:5, [2, 4, 6, 8]]
        RPM_X = select_seq[:, :-1]
        RPM_Y = select_seq[:, 1:]
        # select_seq = RPM[:5, [1, 3, 5]]
        # RPM_X = select_seq[:, :]
        # RPM_Y = select_seq[:, :]

        RPM_X_tensor = torch.Tensor(RPM_X)
        RPM_Y_tensor = torch.Tensor(RPM_Y)
        RPM_X_tensor.cuda()
        RPM_Y_tensor.cuda()

        invol = batch.clone().detach()
        bat_pred, DVF = self.model(invol, rpm_x=RPM_X_tensor, rpm_y=RPM_Y_tensor, future_seq=self.seq) 

#-----------------org No 2d timeseries---------------------------
        # b, seq, c, vol, h, w = batch.size()
        #     # batch.shape = b, seq, c, vol, h, w
        #     # save batch img
        #     # Batch=batch[0,0,0,...]
        #     # # dvf=dvf.permute(1,2,0)
        #     # Batch=Batch.cpu().detach().numpy()
        #     # plt.imshow(Batch)
        #     # plt.show()
        #     # plt.savefig('/workspace/SeqX2Y_PyTorch/test/Imageresult/Batch.png')

        #     #! RPM Bug logging.info("Patient index: %s, RPM index: %s" % (batch_idx, rpm))
        #     # logging.info("Patient index: %s" % (batch_idx))
        # rpm = int(np.random.randint(0, 20, 1))

        # RPM = np.array(self.data)
        # RPM = np.float32(RPM)
        # test_RPM = RPM

        #     # load rpm
        #     # test_rpm_ = test_RPM[rpm,:]
        #     # test_x_rpm = test_RPM[rpm,:1]
        #     # test_x_rpm = np.expand_dims(test_x_rpm,0)
        #     # test_y_rpm = test_RPM[rpm,0:]
        #     # test_y_rpm = np.expand_dims(test_y_rpm,0)

        # # TODO you should fix this, mapping with your data.
        # # ! fake data 
        # test_x_rpm = np.random.rand(1, 7) # patient index, seq
        # test_y_rpm = np.random.rand(1, 7) # same to the seq
        #     # test_x_rpm *= 10
        #     # test_y_rpm *= 10

        #     # invol = torch.Tensor(test_x_)
        #     # invol = invol.permute(0, 1, 5, 2, 3, 4)
        #     # invol = invol.to(device)
        #     # invol = batch.unsqueeze(dim=2)  # b, seq, c, vol, h, w
        # invol = batch.clone().detach()

        # test_x_rpm_tensor = torch.Tensor(test_x_rpm)
        # test_y_rpm_tensor = torch.Tensor(test_y_rpm)
        # test_x_rpm_tensor.cuda()
        # test_y_rpm_tensor.cuda()

        #     # pred the video frames
        #     # invol: 1, 1, 1, 128, 128, 128 # b, seq, c, vol, h, w
        #     # rpm_x: 1, 1
        #     # rpm_y: 1, 9
        # bat_pred, DVF = self.model(invol, rpm_x=test_x_rpm_tensor, rpm_y=test_y_rpm_tensor, future_seq=self.seq) 
#-----------------org No 2d timeseries---------------------------

        # Calc Loss 
        phase_mse_loss_list = []
        phase_smooth_l1_loss_list = []

        # # Origin Loss Function
        # for phase in range(self.seq):
        #     # MSE loss
        #     phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))   # bat_pred[:,:,phase,...].shape => torch.Size([1, 1, 118, 128, 128])
        #     # smooth l1 loss
        #     phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase, ...].expand_as(DVF[:,:,phase,...]))) # DVF[:,:,phase,...].shape => torch.Size([1, 3, 118, 128, 128])                     
        # # sum two loss
        # train_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))

        # chen orign 
        for phase in range(self.seq):
        # for phase in range(self.seq-1):
            phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))   # DVF torch.Size([1, 3, 3, 70, 120, 140])
            phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase, ...].expand_as(DVF[:,:,phase,...]))) # DVF[:,:,phase,...] torch.Size([1, 3, 70, 120, 140])          
            #!FIXME Metrics Test But Erro ValueError: Expected both prediction and target to be 1D or 2D tensors, but received tensors with dimension torch.Size([1, 1, 118, 128, 128])
            # mse_value = self.mse(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...]))
            # mae_value = self.mae(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...]))
            #r2_value = self.r2_score(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...]))                
        train_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))

        # 1 3 5 7 --> 2 4 6 8 
        # for phase in range(self.seq):
        # for phase in range(self.seq-4):
        #     # +1 表示让预测生成的肺与后一个肺做loss
        #     phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:, phase*2+1, ...].expand_as(bat_pred[:,:,phase,...]))) # bat_pred(1,1,3,128,128,128), batch torch.Size([1, 4, 70, 120, 140])
        #     phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase*2+1, ...].expand_as(DVF[:,:,phase,...]))) # DVF[:,:,phase,...] torch.Size([1, 3, 70, 120, 140])              
        # train_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))

        # # Right 1 3 5  --> 2 4 6 
        # for phase in range(0, batch.shape[1], 2): 
        #     # +1 loss was made between the predicted lung and the lung at t+1 time
        #     phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase//2,...], batch[:, phase+1, ...].expand_as(bat_pred[:,:,phase//2,...])))     # bat_pred(1,1,3,128,128,128), batch torch.Size([1, 4, 70, 120, 140])
        #     phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase//2,...], batch[:, phase+1, ...].expand_as(DVF[:,:,phase//2,...])))   # DVF[:,:,phase,...] torch.Size([1, 3, 70, 120, 140])              
        # train_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))
        
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

        # ouyangV1 add spatial transform
        # Transform = Warp(size1=128, size2=128, size3=128).cuda() # spatial transform 
        # for phase in range(self.seq):
        #     T = Transform(bat_pred[:,:,phase,...], batch[:, 0, ...].expand_as(bat_pred[:,:,0,...]))
        #     phase_mse_loss_list.append(F.mse_loss(T, batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))
        #     phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase, ...].expand_as(DVF[:,:,phase,...])))
        # train_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))
        # self.log('train_loss', train_loss)
        # logging.info('train_loss: %d' % train_loss)

        # ouyangV2 add gradient
        # for phase in range(self.seq):
        #     phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))
        #     # phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase, ...].expand_as(DVF[:,:,phase,...])))
        #     input_tensor = batch[:, phase, ...].expand_as(DVF[:,:,phase,...])
        #     input_tensor.requires_grad = True
        #     gradient_phi_t = torch.autograd.grad(outputs=DVF[:,:,phase,...], inputs=input_tensor, grad_outputs=torch.ones_like(DVF[:,:,phase,...]), create_graph=True)[0]
        #     part2_loss = torch.sum(gradient_phi_t.pow(2))
        # train_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(part2_loss, dim=0))

        # ouyangV3 smoothness loss
        # for phase in range(self.seq):
        #     phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))
        #     phase_smooth_l1_loss_list.append(self.calculate_smoothness_loss(DVF[:,:,phase,...]))
        # train_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))
        # self.log('train_loss', train_loss)
        # logging.info('train_loss: %d' % train_loss)

        return train_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        '''
        val step when trainer.fit called.

        Args:
            batch (torch.Tensor): b, seq, vol, c, h, w
            batch_idx (int): batch index, or patient index

        Returns: None
        '''

#------------------------- New ---------------------------------
        b, seq, c, vol, h, w = batch.size()
        # Reading RPM #
        with open('/home/ec2-user/SeqX2Y_PyTorch/dataset/Diagram_Coordinates/1D_rpm.csv', 'r', encoding='utf-8-sig') as f:  # 使用'utf-8-sig'来自动处理BOM
            data = list(csv.reader(f, delimiter=","))        
        RPM = np.array(data)
        RPM = np.float32(RPM)
        
        # When validation, use last patient (5)
        select_seq = RPM[:, [2, 4, 6, 8]] 
        RPM_X = select_seq[:, :-1]
        RPM_Y = select_seq[:, 1:]
        # select_seq = RPM[:5, [1, 3, 5]]
        # RPM_X = select_seq[:, :]
        # RPM_Y = select_seq[:, :]

        RPM_X_tensor = torch.Tensor(RPM_X)
        RPM_Y_tensor = torch.Tensor(RPM_Y)
        RPM_X_tensor.cuda()
        RPM_Y_tensor.cuda()

        invol = batch.clone().detach()

#-----------------org No 2d timeseries---------------------------        
        # b, seq, c, vol, h, w = batch.size()
        # rpm = int(np.random.randint(0, 20, 1))
        #     # logging.info("Patient index: %s, RPM index: %s" % (batch_idx, rpm))
        #     # logging.info("Patient index: %s" % (batch_idx))

        # RPM = np.array(self.data)
        # RPM = np.float32(RPM)
        # test_RPM = RPM

        #     # ! TODO you should fix this, mapping with your data.
        #     # load rpm
        #     # test_rpm_ = test_RPM[rpm,:]
        #     # test_x_rpm = test_RPM[rpm,:1]
        #     # test_x_rpm = np.expand_dims(test_x_rpm,0)
        #     # test_y_rpm = test_RPM[rpm,0:]
        #     # test_y_rpm = np.expand_dims(test_y_rpm,0)

        #     # ! fake data
        #     # test_x_rpm = np.random.rand(1, 10)[0,:9] # patient index, seq
        #     # test_y_rpm = np.random.rand(1, 10)[0,1:]
        # test_x_rpm = np.random.rand(1, 7) #!patient index, seq
        # test_y_rpm = np.random.rand(1, 7) # same to the saeq
        #     # test_x_rpm *= 10
        #     # test_y_rpm *= 10

        #     # invol = torch.Tensor(test_x_)
        #     # invol = invol.permute(0, 1, 5, 2, 3, 4)
        #     # invol = invol.to(device)
        #     # invol = batch.unsqueeze(dim=2) # b, seq, c, vol, h, w
        # invol = batch.clone().detach()
        
        #     # ! TODO you should decrease the seq_len, to reduce the memory usage.
        #     # new_invol = batch[:, :self.seq, ...]

        # test_x_rpm_tensor = torch.Tensor(test_x_rpm)
        # test_y_rpm_tensor = torch.Tensor(test_y_rpm)
        # test_x_rpm_tensor.cuda()
        # test_y_rpm_tensor.cuda()
#-----------------org No 2d timeseries---------------------------  

        # pred the video frames
        with torch.no_grad():
            # invol: 1, 9, 1, 128, 128, 128 # b, seq, c, vol, h, w
            # rpm_x: 1, 1
            # rpm_y: 1, 9
            bat_pred, DVF = self.model(invol, rpm_x=RPM_X_tensor, rpm_y=RPM_Y_tensor, future_seq=self.seq)  # [1,2,3,176,176]
            # bat_pred.shape=(1,1,3,128,128,128) DVF.shape=(1,3,3,128,128,128) 

        # Save images
        # save_dvf_image(DVF, batch_idx, '/workspace/SeqX2Y_PyTorch/test/Imageresult')
        # save_bat_pred_image(bat_pred, batch_idx, '/workspace/SeqX2Y_PyTorch/test/Imageresult')
        save_sitk_images(bat_pred, batch_idx, '/home/ec2-user/SeqX2Y_PyTorch/test/Imageresult')
        save_sitk_DVF_images(DVF, batch_idx, '/home/ec2-user/SeqX2Y_PyTorch/test/Imageresult')

        # calc loss 
        phase_mse_loss_list = []
        phase_smooth_l1_loss_list = []
        # SSIM
        ssim_values = []
        # ssim = SSIM().to(device=1) # data_range = 2
        ssim = SSIM().to(device=0) # AWS = 0
        # NCC
        ncc_values = []
        # DICE
        dice_values = []
        # MAE
        mae_values = []
        # PSNR
        psnr_values = []

        # # Chen+SSIM+NCC+DICE
        # # for phase in range(self.seq):
        # # for phase in range(self.seq-4):
        # for phase in range(0, batch.shape[1], 2):
        #     phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase//2,...], batch[:, phase+1, ...].expand_as(bat_pred[:, : , phase//2, ...])))  # DVF torch.Size([1, 3, 3, 70, 120, 140]), batch torch.Size([1, 4, 70, 120, 140])
        #     phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase//2,...], batch[:, phase+1, ...].expand_as(DVF[:, :, phase//2, ...]))) # but DVF[:,:,phase,...] torch.Size([1, 3, 70, 120, 140])
        #     # ssim: all dimensions together
        #     ssim_value = ssim(bat_pred[:,:,phase//2,...], batch[:, phase+1, ...].expand_as(bat_pred[:, :, phase//2,...]))
        #     ssim_values.append(ssim_value.item())
        #     # # ssim: depth dimension only
        #     # ssim_value = self.calculate_ssim(bat_pred[:,:,phase//2,...], batch[:, phase+1, ...].expand_as(bat_pred[:, :, phase//2,...]))
        #     # ssim_values.append(ssim_value)

        #     # # ncc: all dimensions together
        #     # ncc_value = self.normalized_cross_correlation(bat_pred[:,:,phase//2,...], batch[:,phase+1,...].expand_as(bat_pred[:,:,phase//2,...]))
        #     # ncc_values.append(ncc_value.item())
        #     # ncc:depth dimension only
        #     ncc_value = self.normalized_cross_correlation(bat_pred[:,:,phase//2,...], batch[:,phase+1,...].expand_as(bat_pred[:,:,phase//2,...]))
        #     ncc_values.append(ncc_value)
        #     # dice
        #     dice_value = self.dice_coefficient(bat_pred[:,:,phase//2,...], batch[:,phase+1,...].expand_as(bat_pred[:,:,phase//2,...]))
        #     dice_values.append(dice_value)
        # val_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))

        # Orign Chen+SSIM+NCC+DICE
        for phase in range(self.seq):
            phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))   # DVF torch.Size([1, 3, 3, 70, 120, 140])
            phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase, ...].expand_as(DVF[:,:,phase,...])))
            # ssim
            ssim_value = ssim(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:, :, phase,...]))
            ssim_values.append(ssim_value.item())
            # ssim_values.append(ssim_value)
            # ncc
            ncc_value = self.normalized_cross_correlation(bat_pred[:,:,phase,...], batch[:,phase,...].expand_as(bat_pred[:,:,phase,...]))
            # ncc_values.append(ncc_value.item())
            ncc_values.append(ncc_value)
            # dice
            dice_value = self.dice_coefficient(bat_pred[:,:,phase,...], batch[:,phase,...].expand_as(bat_pred[:,:,phase,...]))
            dice_values.append(dice_value)
            # MAE
            mae = torch.mean(torch.abs(bat_pred[:,:,phase,...] - batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))
            mae_values.append(mae.item())
            # psnr
            psnr_value = psnr(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...]))
            psnr_values.append(psnr_value.item())

        val_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))

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
        # PSNR
        average_psnr = sum(psnr_values) / len(psnr_values) 
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
        self.log('Average PSNR', average_psnr)
        logging.info('Average PSNR: %.4f' % average_psnr)

        # #Draw the image
        # metrics = ['SSIM', 'NCC', 'DICE']
        # # average_dice_cpu = average_dice.cpu().item()
        # values = [average_ssim, average_ncc, average_dice]  # 使用 .item() 转换 PyTorch 张量为 Python 数字
        # # STYLE 2
        # plt.style.use('ggplot')
        # # 创建一个条形图
        # fig, ax = plt.subplots(figsize=(10, 5))  # 可以调整大小以适应您的需求
        # # 绘制条形图
        # bars = ax.bar(metrics, values, color=['salmon', 'cornflowerblue', 'teal'], width=0.5, edgecolor='black', linewidth = 0)
        # # 添加数值标签
        # for bar in bars:
        #     yval = bar.get_height()
        #     plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 4), va='bottom', ha='center')
        # # 设置标题和标签
        # ax.set_title('Average Metric Values', fontsize=16)
        # ax.set_xlabel('Metrics', fontsize=14)
        # ax.set_ylabel('Values', fontsize=14)
        # # 设置 y 轴的限制
        # ax.set_ylim(0, 1)
        # # 移除顶部和右侧的边框线
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # # 显示网格线
        # ax.yaxis.grid(True)
        # # 设置 y 轴刻度标签的大小
        # ax.tick_params(axis='y', labelsize=12)
        # # 显示图表
        # plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
        # plt.show()
        # plt.savefig('/workspace/SeqX2Y_PyTorch/test/Imageresult/matplot2.png')

        # # STYLE 3 heatmap
        # # 创建一个单行的矩阵，每个指标一个值
        # heatmap_data = np.array(values).reshape(1, len(values))
        # # 使用 seaborn 创建热力图
        # plt.figure(figsize=(8, 2))
        # sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap='coolwarm', xticklabels=metrics, yticklabels=False)
        # plt.title('Metric Heatmap')
        # plt.savefig('/workspace/SeqX2Y_PyTorch/test/Imageresult/matplot3.png')
        # plt.show()

        # # chen ORIGIN
        # for phase in range(self.seq):
        # # for phase in range(self.seq-1):
        #     phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:,phase,...].expand_as(bat_pred[:,:,phase,...])))  # DVF torch.Size([1, 3, 3, 70, 120, 140]), batch torch.Size([1, 4, 70, 120, 140])
        #     phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase, ...].expand_as(DVF[:,:,phase,...]))) # but DVF[:,:,phase,...] torch.Size([1, 3, 70, 120, 140])
        # val_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))

        # self.log('val_loss', val_loss)
        # logging.info('val_loss: %.4f' % val_loss)

        # ouyangV1
        # Transform = Warp(size1=128, size2=128, size3=128).cuda() # spatial transform
        # for phase in range(self.seq):
        #     T = Transform(bat_pred[:,:,phase,...], batch[:, 0, ...].expand_as(bat_pred[:,:,0,...]))
        #     phase_mse_loss_list.append(F.mse_loss(T, batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))
        #     phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase, ...].expand_as(DVF[:,:,phase,...])))
        # val_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))
       
        # ||∇ϕt||^2
        # for phase in range(self.seq):
        #     phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))
        #     # phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase, ...].expand_as(DVF[:,:,phase,...])))
        #     DDD=DVF[:,:,phase,...]
        #     DDD.requires_grad = True
        #     gradient_phi_t = torch.autograd.grad(DDD.sum(), DDD, create_graph=True)[0]
        #     part2_loss = torch.sum(gradient_phi_t.pow(2))
        # val_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(part2_loss, dim=0))
        
        # ouyangV3 smoothness loss
        # for phase in range(self.seq):
        #     phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))
        #     phase_smooth_l1_loss_list.append(self.calculate_smoothness_loss(DVF[:,:,phase,...]))
        # val_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))
        # self.log('val_loss', val_loss)
        # logging.info('val_loss: %d' % val_loss)

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
