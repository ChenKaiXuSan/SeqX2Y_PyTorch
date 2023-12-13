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
Last Modified: 2023-10-02 08:14:09
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------
2023-09-26	KX.C	change the train and val process, here we think need use the self.seq to control the seq_len, to reduce the memory usage.

'''

# %%
import os, csv, logging, shutil
import seaborn as sns
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
            # ! FIXME
            nf = 48, in_chan=1, size1=self.vol, size2=self.img_size, size3=self.img_size)
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
    
    # Calculate NCC values
    def normalized_cross_correlation(self, x, y):
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        x_normalized = x - mean_x
        y_normalized = y - mean_y
        ncc = torch.sum(x_normalized * y_normalized) / (torch.sqrt(torch.sum(x_normalized ** 2)) * torch.sqrt(torch.sum(y_normalized ** 2)))
        return ncc

    # Calculate Dice values
    def dice_coefficient(self, pred, target):
        smooth = 1.0  # Used to prevent division by zero
        # Binarize the prediction and target, and the threshold is usually set at 0.5
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return dice
    
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

        b, seq, c, vol, h, w = batch.size()

        # save batch img
        # Batch=batch[0,0,0,...]
        # # dvf=dvf.permute(1,2,0)
        # Batch=Batch.cpu().detach().numpy()
        # plt.imshow(Batch)
        # plt.show()
        # plt.savefig('/workspace/SeqX2Y_PyTorch/test/Imageresult/Batch.png')

        rpm = int(np.random.randint(0, 20, 1))
        #! RPM Bug logging.info("Patient index: %s, RPM index: %s" % (batch_idx, rpm))
        # logging.info("Patient index: %s" % (batch_idx))

        RPM = np.array(self.data)
        RPM = np.float32(RPM)
        test_RPM = RPM

        # load rpm
        # test_rpm_ = test_RPM[rpm,:]
        # test_x_rpm = test_RPM[rpm,:1]
        # test_x_rpm = np.expand_dims(test_x_rpm,0)
        # test_y_rpm = test_RPM[rpm,0:]
        # test_y_rpm = np.expand_dims(test_y_rpm,0)

        # TODO you should fix this, mapping with your data.
        # ! fake data 
        test_x_rpm = np.random.rand(1, 7) # patient index, seq
        test_y_rpm = np.random.rand(1, 7) # same to the seq
        # test_x_rpm *= 10
        # test_y_rpm *= 10

        # invol = torch.Tensor(test_x_)
        # invol = invol.permute(0, 1, 5, 2, 3, 4)
        # invol = invol.to(device)
        # invol = batch.unsqueeze(dim=2)  # b, seq, c, vol, h, w
        invol = batch.clone().detach()

        test_x_rpm_tensor = torch.Tensor(test_x_rpm)
        test_y_rpm_tensor = torch.Tensor(test_y_rpm)
        test_x_rpm_tensor.cuda()
        test_y_rpm_tensor.cuda()

        # pred the video frames
        # invol: 1, 1, 1, 128, 128, 128 # b, c, f, vol, h, w
        # rpm_x: 1, 1
        # rpm_y: 1, 9
        bat_pred, DVF = self.model(invol, rpm_x=test_x_rpm_tensor, rpm_y=test_y_rpm_tensor, future_seq=self.seq)  # [1,2,3,176,176]

        # calc loss 
        phase_mse_loss_list = []
        phase_smooth_l1_loss_list = []

        # chen
        # for phase in range(self.seq):
        for phase in range(self.seq-1):
            phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))   # DVF torch.Size([1, 3, 3, 70, 120, 140])
            phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase, ...].expand_as(DVF[:,:,phase,...]))) # DVF[:,:,phase,...] torch.Size([1, 3, 70, 120, 140])          
            #!FIXME Metrics Test But Erro ValueError: Expected both prediction and target to be 1D or 2D tensors, but received tensors with dimension torch.Size([1, 1, 118, 128, 128])
            # mse_value = self.mse(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...]))
            # mae_value = self.mae(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...]))
            #r2_value = self.r2_score(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...]))                
        train_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))
        
        # Storing train loss on the True first iteration 确保只在第一次实际训练迭代时设置初始训练损失
        if not self.initial_train_loss_set:
            self.initial_train_loss = train_loss.detach().clone()
            self.initial_train_loss_set = True
        relative_train_loss = train_loss / self.initial_train_loss
        #save logs
        logging.info("Patient index: %s" % (batch_idx))
        self.log('train_loss', train_loss, on_epoch=True, on_step=True)
        logging.info('train_loss: %.4f' % train_loss)
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
        b, seq, c, vol, h, w = batch.size()

        rpm = int(np.random.randint(0, 20, 1))
        # logging.info("Patient index: %s, RPM index: %s" % (batch_idx, rpm))
        # logging.info("Patient index: %s" % (batch_idx))

        RPM = np.array(self.data)
        RPM = np.float32(RPM)
        test_RPM = RPM

        # ! TODO you should fix this, mapping with your data.
        # load rpm
        # test_rpm_ = test_RPM[rpm,:]
        # test_x_rpm = test_RPM[rpm,:1]
        # test_x_rpm = np.expand_dims(test_x_rpm,0)
        # test_y_rpm = test_RPM[rpm,0:]
        # test_y_rpm = np.expand_dims(test_y_rpm,0)

        # ! fake data
        # test_x_rpm = np.random.rand(1, 10)[0,:9] # patient index, seq
        # test_y_rpm = np.random.rand(1, 10)[0,1:]
        test_x_rpm = np.random.rand(1, 7) #!patient index, seq
        test_y_rpm = np.random.rand(1, 7) # same to the saeq
        # test_x_rpm *= 10
        # test_y_rpm *= 10

        # invol = torch.Tensor(test_x_)
        # invol = invol.permute(0, 1, 5, 2, 3, 4)
        # invol = invol.to(device)
        # invol = batch.unsqueeze(dim=2) # b, seq, c, vol, h, w
        invol = batch.clone().detach()
        
        # ! TODO you should decrease the seq_len, to reduce the memory usage.
        # new_invol = batch[:, :self.seq, ...]

        test_x_rpm_tensor = torch.Tensor(test_x_rpm)
        test_y_rpm_tensor = torch.Tensor(test_y_rpm)
        test_x_rpm_tensor.cuda()
        test_y_rpm_tensor.cuda()

        # pred the video frames
        with torch.no_grad():
            # invol: 1, 9, 1, 128, 128, 128 # b, seq, c, vol, h, w
            # rpm_x: 1, 1
            # rpm_y: 1, 9
            bat_pred, DVF = self.model(invol, rpm_x=test_x_rpm_tensor, rpm_y=test_y_rpm_tensor, future_seq=self.seq)  # [1,2,3,176,176]
            # bat_pred.shape=(1,1,3,128,128,128) DVF.shape=(1,3,3,128,128,128) 

        # save DVF img
        savepath = '/workspace/SeqX2Y_PyTorch/test/Imageresult'

        # make dir 
        save_path = savepath + "/" + "%3.3d" % batch_idx
        if not os.path.exists(save_path):os.makedirs(save_path)

        # save dvf img
        dvf=DVF[0,:,0,0,...]
        dvf=dvf.permute(1,2,0)
        dvf=dvf.cpu().detach().numpy()
        plt.imshow(dvf)
        plt.show()
        plt.savefig('/workspace/SeqX2Y_PyTorch/test/Imageresult/dvf.png')

        # save bat pred
        Bat_Pred=bat_pred[0,0,:,0,...]
        Bat_Pred=Bat_Pred.permute(1,2,0)
        Bat_Pred=Bat_Pred.cpu().detach().numpy()
        # plt.imshow(Bat_Pred)
        # plt.show()
        plt.savefig('/workspace/SeqX2Y_PyTorch/test/Imageresult/Bat_Pred.png')

        # save predict img
        BAT_PRED = bat_pred.cpu().detach().numpy() # 1, 1, future_seq, 128, 128, 128
        BAT_PRED = np.squeeze(BAT_PRED) # pred_feat, 128, 128, 128
        
        writer = sitk.ImageFileWriter()
        pI1, pI2, pI3 = np.squeeze(BAT_PRED[0, ...]), np.squeeze(BAT_PRED[1, ...]), np.squeeze(BAT_PRED[2, ...]) #seq = 3   
        # pI1, pI2, pI3, pI4 = np.squeeze(BAT_PRED[0, ...]), np.squeeze(BAT_PRED[1, ...]), np.squeeze(BAT_PRED[2, ...]), np.squeeze(BAT_PRED[3, ...]) #seq = 4       
        # pI1, pI2, pI3, pI4, pI5, pI6 = np.squeeze(BAT_PRED[0, ...]), np.squeeze(BAT_PRED[1, ...]), np.squeeze(BAT_PRED[2, ...]), np.squeeze(BAT_PRED[3, ...]), np.squeeze(BAT_PRED[4, ...]), np.squeeze(BAT_PRED[5, ...]) #seq = 6      
        # pI1, pI2, pI3, pI4, pI5, pI6, pI7, pI8 = np.squeeze(BAT_PRED[0, ...]), np.squeeze(BAT_PRED[1, ...]), np.squeeze(BAT_PRED[2, ...]), np.squeeze(BAT_PRED[3, ...]), np.squeeze(BAT_PRED[4, ...]), np.squeeze(BAT_PRED[5, ...]), np.squeeze(BAT_PRED[6, ...]), np.squeeze(BAT_PRED[6, ...]) #seq = 7
        writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale1_predict.nrrd")
        writer.Execute(sitk.GetImageFromArray(pI1))
        writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale2_predict.nrrd")
        writer.Execute(sitk.GetImageFromArray(pI2))
        writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale3_predict.nrrd")
        writer.Execute(sitk.GetImageFromArray(pI3))
        # writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale4_predict.nrrd")
        # writer.Execute(sitk.GetImageFromArray(pI4))
        # writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale5_predict.nrrd")
        # writer.Execute(sitk.GetImageFromArray(pI5))
        # writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale6_predict.nrrd")
        # writer.Execute(sitk.GetImageFromArray(pI6))
        # writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale7_predict.nrrd")
        # writer.Execute(sitk.GetImageFromArray(pI7))
        # writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale8_predict.nrrd")
        # writer.Execute(sitk.GetImageFromArray(pI8))
        
        # # Permute DVF & Save DVF
        # def dvf_(d):
        #     x = d[0,...]
        #     x = np.reshape(x, [1,118, 128, 128])
        #     y = d[1,...]
        #     y = np.reshape(y, [1,118, 128, 128])
        #     z = d[2,...]
        #     z = np.reshape(z, [1,118, 128, 128])
        #     out = np.concatenate([z,y,x],0)
        #     return out
        
        # Dvf = DVF.cpu().detach().numpy() # 1,3,9, 128, 128
        # Dvf = np.squeeze(Dvf) # 3, 9, 128, 128, 128
        # DVF2, DVF3, DVF4 = dvf_(Dvf[:,0,...]), dvf_(Dvf[:,1,...]), dvf_(Dvf[:,2,...])

        # writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "DVF2.nrrd")
        # writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF2), [1,2,3,0]))) # 3 1 2
        # writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "DVF3.nrrd")
        # writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF3), [1,2,3,0]))) # 3 1 2
        # writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "DVF4.nrrd")
        # writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF4), [1,2,3,0]))) # 3 1 2

        # calc loss 
        phase_mse_loss_list = []
        phase_smooth_l1_loss_list = []
        # SSIM
        ssim_values = []
        ssim = SSIM().to(device=1)
        # NCC
        ncc_values = []
        # DICE
        dice_values = []

        # Chen+SSIM+NCC+DICE
        # for phase in range(self.seq):
        for phase in range(self.seq-1):
            phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:,phase,...].expand_as(bat_pred[:,:,phase,...])))  # DVF torch.Size([1, 3, 3, 70, 120, 140]), batch torch.Size([1, 4, 70, 120, 140])
            phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase, ...].expand_as(DVF[:,:,phase,...]))) # but DVF[:,:,phase,...] torch.Size([1, 3, 70, 120, 140])
            # ssim
            ssim_value = ssim(bat_pred[:,:,phase,...], batch[:,phase,...].expand_as(bat_pred[:,:,phase,...]))
            ssim_values.append(ssim_value.item())
            # ncc
            ncc_value = self.normalized_cross_correlation(bat_pred[:,:,phase,...], batch[:,phase,...].expand_as(bat_pred[:,:,phase,...]))
            ncc_values.append(ncc_value.item())
            # dice
            dice_value = self.dice_coefficient(bat_pred[:,:,phase,...], batch[:,phase,...].expand_as(bat_pred[:,:,phase,...]))
            dice_values.append(dice_value)

        val_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))
        # Storing val_loss on the True first iteration 确保只在第一次实际验证迭代时设置初始验证损失
        if not self.initial_val_loss_set:
            self.initial_val_loss = val_loss.detach().clone()
            self.initial_val_loss_set = True          
        relative_val_loss = val_loss / self.initial_val_loss 
        # 
        average_ssim = sum(ssim_values) / len(ssim_values)
        average_ncc = sum(ncc_values) / len(ncc_values)
        average_dice = sum(dice_values) / len(dice_values)
        # save logs  
        logging.info("Patient index: %s" % (batch_idx)) 
        self.log('val_loss', val_loss, on_epoch=True, on_step=True)
        logging.info('val_loss: %.4f' % val_loss)
        print("Current val_loss:", val_loss.item())
        # print(f"Average SSIM: {average_ssim}")
        self.log('Average SSIM', average_ssim)
        logging.info('Average SSIM: %.4f' % average_ssim)
        self.log('Average NCC', average_ncc)
        logging.info('Average NCC: %.4f' % average_ncc)
        self.log('Average Dice', average_dice)
        logging.info('Average Dice: %.4f' % average_dice.item())

        # Draw the image
        metrics = ['SSIM', 'NCC', 'DICE']
        average_dice_cpu = average_dice.cpu().item()
        values = [average_ssim, average_ncc, average_dice_cpu]  # 使用 .item() 转换 PyTorch 张量为 Python 数字
        # #  STYLE 1 draw bar picture
        # plt.figure(figsize=(10, 5))
        # plt.bar(metrics, values, color=['blue', 'green', 'red'])
        # plt.title('Average Metric Values')
        # plt.xlabel('Metrics')
        # plt.ylabel('Values')
        # # show value
        # for i, v in enumerate(values):
        #     plt.text(i, v + 0.01, "{:.4f}".format(v), ha='center', va='bottom')
        # plt.show()
        # plt.savefig('/workspace/SeqX2Y_PyTorch/test/Imageresult/matplot1.png')

        # STYLE 2
        plt.style.use('ggplot')
        # 创建一个条形图
        fig, ax = plt.subplots(figsize=(10, 5))  # 可以调整大小以适应您的需求
        # 绘制条形图
        bars = ax.bar(metrics, values, color=['salmon', 'cornflowerblue', 'teal'], width=0.5, edgecolor='black', linewidth = 0)
        # 添加数值标签
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 4), va='bottom', ha='center')
        # 设置标题和标签
        ax.set_title('Average Metric Values', fontsize=16)
        ax.set_xlabel('Metrics', fontsize=14)
        ax.set_ylabel('Values', fontsize=14)
        # 设置 y 轴的限制
        ax.set_ylim(0, 1)
        # 移除顶部和右侧的边框线
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # 显示网格线
        ax.yaxis.grid(True)
        # 设置 y 轴刻度标签的大小
        ax.tick_params(axis='y', labelsize=12)
        # 显示图表
        plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
        plt.show()
        plt.savefig('/workspace/SeqX2Y_PyTorch/test/Imageresult/matplot2.png')

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
