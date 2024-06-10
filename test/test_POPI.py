#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
！！！！！Attention：if use the LUNA.npz file, you should make the deepth be 128 not 118
File: /workspace/SeqX2Y_PyTorch/test/test_POPI.py
Project: /workspace/SeqX2Y_PyTorch/test
Created Date: Wednesday January 17th 2024
Author: Hao Ouyang
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday January 17th 2024 10:52:28 pm
Modified By: the developer formerly known as Hao Ouyang at <ouyanghaomail@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
import os, logging, csv, warnings, sys
from omegaconf import DictConfig
# add path to system.
# sys.path.append('/workspace/SeqX2Y_PyTorch')
sys.path.append('/home/ec2-user/SeqX2Y_PyTorch')
# ignore warnings
warnings.filterwarnings('ignore')

import numpy as np
import random as rn

import torch
import torch.nn as nn

# from project.models.seq2seq_4DCT_voxelmorph import EncoderDecoderConvLSTM
from project.models.Time_series_seq2seq_4DCT_voxelmorph import EncoderDecoderConvLSTM
from project.models.unet_model import Unet
from project.models.Warp import Warp
from project.dataloader.ct_dataset import CTDataset
from torch.utils.data import DataLoader
from project.dataloader.data_loader import CT_normalize
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

#from models.seq2seq_ConvLSTM3d import EncoderDecoderConvLSTM
import matplotlib.pyplot as plt

import SimpleITK as sitk

# config loader
import hydra

#from scipy.ndimage import zoom

@hydra.main(version_base=None, config_path="/home/ec2-user/SeqX2Y_PyTorch/configs", config_name="config.yaml")
def main(config: DictConfig):

    #Loading image # 
    Data = np.load(config.test['data'])['Data']
    #Loading Lung Mask#
    Seq = np.load(config.test['mask'])['Data']

    # Crop data #
    test_x = Data[:,16:144,16:144,16:144,:]
    test_sx = Seq[:,16:144,16:144,16:144,:]

    # check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    Hidden_dim=96

    # load train process
    ConvLSTMmodel = EncoderDecoderConvLSTM(nf=Hidden_dim, in_chan=1, size1=128, size2=128, size3=128)
    # ConvLSTMmodel.to(device)

    # ckpt 
    # ckpt = torch.load(config.test['ckpt'])['state_dict']
    ckpt = torch.load(config.test['ckpt'])['state_dict']

    # here we need delete the 'model' from the loaded ckpt
    # because the pytorch lightning method will save the whole keyword into the ckpt file.
    new_ckpt = {}
    for k, v in ckpt.items():
        new_ckpt['.'.join(k.split('.')[1:])] = v

    ConvLSTMmodel.load_state_dict(new_ckpt)
    ConvLSTMmodel.to(device=1)
    # 打印模型结构
    print(ConvLSTMmodel)
    # Transform = Warp(size1=128, size2=128, size3=128)
    # print(Transform)
    # Transform.to(device)

    # Reading RPM #
    with open(config.test['rpm'], 'r') as f:
        data = list(csv.reader(f, delimiter=","))
    
    RPM = np.array(data)
    RPM = np.float32(RPM)
    test_RPM = RPM

    sample_info_dict = CTDataset(
                # data_path="/workspace/data/POPI_valdata3",
                # data_path2D="/workspace/data/POPI_val2D_seq3",               
                ct_transform= Compose(
                                        [
                                            CT_normalize(128),
                                        ]
                                    ),
                time_series_transform= Compose(
                                        [
                                            ToTensor(),
                                            Resize(size=[128, 128]),
                                            Normalize((0.45), (0.225)),
                                            lambda x: x/255.0,
                                        ]
                                    ),
                vol=118,
            )
    
    dataloader = DataLoader(
            sample_info_dict,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )

    for num, batch in enumerate(dataloader):
        print(num)
        print(batch)
        ct_data = batch['4DCT'].to(device=1)
        time_series_img = batch['2D_time_series'].to(device=1)
        INVOL = ct_data.clone().detach()
    
    # Generate simulation for Each POPI data #
    for i in range(0,3):
        patient = i 
        # Randomly choose RPM #
        rpm = int(np.random.randint(0, 20, 1))
        logging.info("Patient index: %s, RPM index: %s" % (patient, rpm))
        test_x_ = test_x[patient,...]
        test_x_ = np.expand_dims(test_x_, 0)
        test_x_ = np.expand_dims(test_x_, 0) # 1,1,128,128,128,1

        # test_sx_ = test_sx[patient, ...]
        # test_sx_ = np.expand_dims(test_sx_, 0)
        # test_sx_ = torch.Tensor(test_sx_)
        # test_sx_ = test_sx_.permute(0,4,1,2,3)
        # test_sx_ = test_sx_.to(device)

        test_rpm_ = test_RPM[rpm,:]
        test_x_rpm = test_RPM[rpm,:1]
        test_x_rpm = np.expand_dims(test_x_rpm,0)
        test_y_rpm = test_RPM[rpm,0:]
        test_y_rpm = np.expand_dims(test_y_rpm,0)

        invol = torch.Tensor(test_x_)
        invol = invol.permute(0, 1, 5, 2, 3, 4)
        # invol = invol.to(device)
        # invol = invol.clone().detach()
        invol = invol.to(device=1)
        test_x_rpm_tensor = torch.Tensor(test_x_rpm)
        test_y_rpm_tensor = torch.Tensor(test_y_rpm)
        test_x_rpm_tensor = test_x_rpm_tensor.to(device=1)
        test_y_rpm_tensor = test_y_rpm_tensor.to(device=1)
        # test_x_rpm_tensor.to(device)
        # test_y_rpm_tensor.to(device)
        # test_x_rpm_tensor = test_x_rpm_tensor.cuda()
        # test_y_rpm_tensor = test_y_rpm_tensor.cuda()

        # Prediction, set to eval.
        ConvLSTMmodel.eval()
        # invol: 1, 1, 1, 128, 128, 128
        # rpm_x: 1, 1
        # rpm_y: 1, 9
        with torch.no_grad():
            bat_pred, DVF = ConvLSTMmodel(invol, time_series_img, future_seq=3)  # [1,2,3,176,176]

        #bat_pred = bat_pred.cpu().detach().numpy()
        #DVF = DVF.cpu().detach().numpy()
        #bat_pred = np.squeeze(bat_pred)

        # todo
        # Contour propagation #
        # S2, S3, S4 = Transform(test_sx_, DVF[:,:,0,...]),Transform(test_sx_, DVF[:,:,1,...]),Transform(test_sx_, DVF[:,:,2,...])
        # S5, S6, S7 = Transform(test_sx_, DVF[:,:,3,...]),Transform(test_sx_, DVF[:,:,4,...]),Transform(test_sx_, DVF[:,:,5,...])
        # S8, S9, S10 = Transform(test_sx_, DVF[:,:,6,...]),Transform(test_sx_, DVF[:,:,7,...]),Transform(test_sx_, DVF[:,:,8,...])

        # S2, S3, S4 = S2.cpu().detach().numpy(), S3.cpu().detach().numpy(), S4.cpu().detach().numpy()
        # S5, S6, S7 = S5.cpu().detach().numpy(), S6.cpu().detach().numpy(), S7.cpu().detach().numpy()
        # S8, S9, S10 = S8.cpu().detach().numpy(), S9.cpu().detach().numpy(), S10.cpu().detach().numpy()

        bat_pred = bat_pred.cpu().detach().numpy() # 1, 1, pred_feat, 128, 128, 128
        # bat_pred=(1, 1, 9, 128, 128, 128)
        DVF = DVF.cpu().detach().numpy() #1,3,9, 128, 128
        # DVF=(1, 3, 9, 128, 128, 128)
        bat_pred = np.squeeze(bat_pred) # pred_feat, 128, 128, 128
        # bat_pred=(9, 128, 128, 128)
        DVF = np.squeeze(DVF) # 3, 9, 128, 128, 128
        # DVF=(3, 9, 128, 128, 128)

        I1 = np.squeeze(test_x_[:,0, ...]) # 128, 128, 128
        #ex = np.squeeze(test_x2_)

        # D2, D3, D4, D5 = DVF[:,0,...], DVF[:,1,...], DVF[:,2,...], DVF[:,3,...]
        # D6, D7, D8, D9, D10 = DVF[:,4,...], DVF[:,5,...], DVF[:,6,...], DVF[:,7,...],DVF[:,8,...]
        # D2-D8=(3, 128, 128, 128)

        pI2, pI3, pI4 = np.squeeze(bat_pred[0, ...]), np.squeeze(bat_pred[1, ...]), np.squeeze(bat_pred[2, ...])
        # pI5, pI6, pI7 = np.squeeze(bat_pred[3, ...]), np.squeeze(bat_pred[4, ...]), np.squeeze(bat_pred[5, ...])
        # pI8, pI9, pI10 = np.squeeze(bat_pred[6,...]), np.squeeze(bat_pred[7,...]), np.squeeze(bat_pred[8,...])
        # pI2-pI10=(128,128,128)

        # Save results #
        savepath = config.test.log_path

        if not os.path.exists(savepath + "/" + "%3.3d" % patient):
            os.makedirs(savepath + "/" + "%3.3d" % patient)

        np.savetxt(savepath + "/" + "%3.3d" % patient + "/" +  'test_rpm.csv', test_rpm_, fmt="%1.4f", delimiter=",")
        writer = sitk.ImageFileWriter()
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale.nrrd")
        writer.Execute(sitk.GetImageFromArray(I1))

        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale2_predict.nrrd")
        writer.Execute(sitk.GetImageFromArray(pI2))
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale3_predict.nrrd")
        writer.Execute(sitk.GetImageFromArray(pI3))
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale4_predict.nrrd")
        writer.Execute(sitk.GetImageFromArray(pI4))
        # writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale5_predict.nrrd")
        # writer.Execute(sitk.GetImageFromArray(pI5))
        # writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale6_predict.nrrd")
        # writer.Execute(sitk.GetImageFromArray(pI6))
        # writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale7_predict.nrrd")
        # writer.Execute(sitk.GetImageFromArray(pI7))
        # writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale8_predict.nrrd")
        # writer.Execute(sitk.GetImageFromArray(pI8))
        # writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale9_predict.nrrd")
        # writer.Execute(sitk.GetImageFromArray(pI9))
        # writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale10_predict.nrrd")
        # writer.Execute(sitk.GetImageFromArray(pI10))

        # Permute DVF #
        def dvf_(d):
            x = d[0,...]
            x = np.reshape(x, [1,128, 128, 128])
            y = d[1,...]
            y = np.reshape(y, [1,128, 128, 128])
            z = d[2,...]
            z = np.reshape(z, [1,128, 128, 128])
            out = np.concatenate([z,y,x],0)
            return out

        DVF2, DVF3, DVF4 = dvf_(DVF[:,0,...]), dvf_(DVF[:,1,...]), dvf_(DVF[:,2,...])
        # DVF2, DVF3, DVF4, DVF5 = dvf_(DVF[:,0,...]), dvf_(DVF[:,1,...]), dvf_(DVF[:,2,...]), dvf_(DVF[:,3,...])
        # DVF6, DVF7, DVF8, DVF9, DVF10 = dvf_(DVF[:,4,...]), dvf_(DVF[:,5,...]), dvf_(DVF[:,6,...]), dvf_(DVF[:,7,...]),dvf_(DVF[:,8,...])


        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "DVF2.nrrd")
        writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF2), [1,2,3,0]))) # 3 1 2
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "DVF3.nrrd")
        writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF3), [1,2,3,0]))) # 3 1 2
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "DVF4.nrrd")
        writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF4), [1,2,3,0]))) # 3 1 2
        # writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "DVF5.nrrd")
        # writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF5), [1,2,3,0]))) # 3 1 2
        # writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "DVF6.nrrd")
        # writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF6), [1,2,3,0]))) # 3 1 2
        # writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "DVF7.nrrd")
        # writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF7), [1,2,3,0]))) # 3 1 2
        # writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "DVF8.nrrd")
        # writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF8), [1,2,3,0]))) # 3 1 2
        # writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "DVF9.nrrd")
        # writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF9), [1,2,3,0]))) # 3 1 2
        # writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "DVF10.nrrd")
        # writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF10), [1,2,3,0]))) # 3 1 2

if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()