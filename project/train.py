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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from pytorch_lightning import LightningModule
from torchmetrics import classification

from models.seq2seq_4DCT_voxelmorph import EncoderDecoderConvLSTM
from models.Warp import Warp

# %%
class PredictLightningModule(LightningModule):

    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.img_size = hparams.data.img_size
        self.lr = hparams.optimizer.lr
        self.seq = hparams.train.seq
        self.vol = hparams.train.vol

        self.model = EncoderDecoderConvLSTM(
            # nf=96, in_chan=1, size1=30, size2=176, size3=140)
            #  nf=96, in_chan=1, size1=70, size2=120, size3=140)
            # ! FIXME
            nf = 96, in_chan=1, size1=self.vol, size2=self.img_size, size3=self.img_size)
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
        logging.info("Patient index: %s, RPM index: %s" % (batch_idx, rpm))

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
        test_x_rpm = np.random.rand(1, 4) # patient index, seq
        test_y_rpm = np.random.rand(1, 4) 
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
        for phase in range(self.seq):
            phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))   # DVF torch.Size([1, 3, 3, 70, 120, 140])
            phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase, ...].expand_as(DVF[:,:,phase,...]))) # DVF[:,:,phase,...] torch.Size([1, 3, 70, 120, 140])
        train_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))
        self.log('train_loss', train_loss)
        logging.info('train_loss: %.4f' % train_loss)

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
        logging.info("Patient index: %s, RPM index: %s" % (batch_idx, rpm))

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
        test_x_rpm = np.random.rand(1, 4) # patient index, seq
        test_y_rpm = np.random.rand(1, 4)
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
        plt.imshow(Bat_Pred*255)
        plt.show()
        plt.savefig('/workspace/SeqX2Y_PyTorch/test/Imageresult/Bat_Pred.png')

        # save predict img
        BAT_PRED = bat_pred.cpu().detach().numpy() # 1, 1, future_seq, 128, 128, 128
        BAT_PRED = np.squeeze(BAT_PRED) # pred_feat, 128, 128, 128
        
        writer = sitk.ImageFileWriter()
        pI2, pI3, pI4 = np.squeeze(BAT_PRED[0, ...]), np.squeeze(BAT_PRED[1, ...]), np.squeeze(BAT_PRED[2, ...])       
        writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale2_predict.nrrd")
        writer.Execute(sitk.GetImageFromArray(pI2))
        writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale3_predict.nrrd")
        writer.Execute(sitk.GetImageFromArray(pI3))
        writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale4_predict.nrrd")
        writer.Execute(sitk.GetImageFromArray(pI4))
        writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale5_predict.nrrd")

        # Permute DVF & Save DVF
        def dvf_(d):
            x = d[0,...]
            x = np.reshape(x, [1,128, 128, 128])
            y = d[1,...]
            y = np.reshape(y, [1,128, 128, 128])
            z = d[2,...]
            z = np.reshape(z, [1,128, 128, 128])
            out = np.concatenate([z,y,x],0)
            return out
        
        Dvf = DVF.cpu().detach().numpy() # 1,3,9, 128, 128
        Dvf = np.squeeze(Dvf) # 3, 9, 128, 128, 128
        DVF2, DVF3, DVF4 = dvf_(Dvf[:,0,...]), dvf_(Dvf[:,1,...]), dvf_(Dvf[:,2,...])

        writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "DVF2.nrrd")
        writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF2), [1,2,3,0]))) # 3 1 2
        writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "DVF3.nrrd")
        writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF3), [1,2,3,0]))) # 3 1 2
        writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "DVF4.nrrd")
        writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF4), [1,2,3,0]))) # 3 1 2

        # calc loss 
        phase_mse_loss_list = []
        phase_smooth_l1_loss_list = []

        # chen
        for phase in range(self.seq):
            phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:,phase,...].expand_as(bat_pred[:,:,phase,...])))  # DVF torch.Size([1, 3, 3, 70, 120, 140]), batch torch.Size([1, 4, 70, 120, 140])
            phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase, ...].expand_as(DVF[:,:,phase,...]))) # but DVF[:,:,phase,...] torch.Size([1, 3, 70, 120, 140])
        val_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))

        self.log('val_loss', val_loss)
        logging.info('val_loss: %.4f' % val_loss)

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
