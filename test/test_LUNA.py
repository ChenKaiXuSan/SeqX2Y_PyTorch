import os, logging, csv, warnings, sys
from omegaconf import DictConfig
# add path to system.
sys.path.append('/workspace/SeqX2Y_PyTorch')
# ignore warnings
warnings.filterwarnings('ignore')

import numpy as np
import random as rn

import torch
import torch.nn as nn

from project.models.seq2seq_4DCT_voxelmorph import EncoderDecoderConvLSTM
from project.models.unet_model import Unet
from project.models.Warp import Warp

#from models.seq2seq_ConvLSTM3d import EncoderDecoderConvLSTM
import matplotlib.pyplot as plt

import SimpleITK as sitk

# config loader
import hydra

#from scipy.ndimage import zoom

@hydra.main(version_base=None, config_path="/workspace/SeqX2Y_PyTorch/configs", config_name="config.yaml")
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
    ConvLSTMmodel = EncoderDecoderConvLSTM(nf=Hidden_dim, in_chan=1, size1=128, size2=128, size3=128)
    ConvLSTMmodel.to(device)
    ConvLSTMmodel.load_state_dict(torch.load(config.test['ckpt']))
    Transform = Warp(size1=128, size2=128, size3=128)
    Transform.to(device)

    # Reading RPM #
    with open(config.test['rpm'], 'r') as f:
        data = list(csv.reader(f, delimiter=","))
    
    RPM = np.array(data)
    RPM = np.float32(RPM)
    test_RPM = RPM

    # Generate simulation for Each LUNA data #
    for i in range(0,10):
        patient = i 
        # Randomly choose RPM #
        rpm = int(np.random.randint(0, 20, 1))
        logging.info("Patient index: %s, RPM index: %s" % (patient, rpm))
        test_x_ = test_x[patient,...]
        test_x_ = np.expand_dims(test_x_, 0)
        test_x_ = np.expand_dims(test_x_, 0) # 1,1,128,128,128,1

        test_sx_ = test_sx[patient, ...]
        test_sx_ = np.expand_dims(test_sx_, 0)
        test_sx_ = torch.Tensor(test_sx_)
        test_sx_ = test_sx_.permute(0,4,1,2,3)
        test_sx_ = test_sx_.to(device)

        test_rpm_ = test_RPM[rpm,:]
        test_x_rpm = test_RPM[rpm,:1]
        test_x_rpm = np.expand_dims(test_x_rpm,0)
        test_y_rpm = test_RPM[rpm,0:]
        test_y_rpm = np.expand_dims(test_y_rpm,0)

        invol = torch.Tensor(test_x_)
        invol = invol.permute(0, 1, 5, 2, 3, 4)
        invol = invol.to(device)
        test_x_rpm_tensor = torch.Tensor(test_x_rpm)
        test_y_rpm_tensor = torch.Tensor(test_y_rpm)
        test_x_rpm_tensor.to(device)
        test_y_rpm_tensor.to(device)

        # Prediction, set to eval.
        ConvLSTMmodel.eval()
        # invol: 1, 1, 1, 128, 128, 128
        # rpm_x: 1, 1
        # rpm_y: 1, 9
        with torch.no_grad():
            bat_pred, DVF = ConvLSTMmodel(invol, rpm_x=test_x_rpm_tensor, rpm_y=test_y_rpm_tensor, future_seq=9)  # [1,2,3,176,176]

        #bat_pred = bat_pred.cpu().detach().numpy()
        #DVF = DVF.cpu().detach().numpy()
        #bat_pred = np.squeeze(bat_pred)

        # todo
        # Contour propagation #
        S2, S3, S4 = Transform(test_sx_, DVF[:,:,0,...]),Transform(test_sx_, DVF[:,:,1,...]),Transform(test_sx_, DVF[:,:,2,...])
        S5, S6, S7 = Transform(test_sx_, DVF[:,:,3,...]),Transform(test_sx_, DVF[:,:,4,...]),Transform(test_sx_, DVF[:,:,5,...])
        S8, S9, S10 = Transform(test_sx_, DVF[:,:,6,...]),Transform(test_sx_, DVF[:,:,7,...]),Transform(test_sx_, DVF[:,:,8,...])

        S2, S3, S4 = S2.cpu().detach().numpy(), S3.cpu().detach().numpy(), S4.cpu().detach().numpy()
        S5, S6, S7 = S5.cpu().detach().numpy(), S6.cpu().detach().numpy(), S7.cpu().detach().numpy()
        S8, S9, S10 = S8.cpu().detach().numpy(), S9.cpu().detach().numpy(), S10.cpu().detach().numpy()

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

        D2, D3, D4, D5 = DVF[:,0,...], DVF[:,1,...], DVF[:,2,...], DVF[:,3,...]
        D6, D7, D8, D9, D10 = DVF[:,4,...], DVF[:,5,...], DVF[:,6,...], DVF[:,7,...],DVF[:,8,...]
        # D2-D8=(3, 128, 128, 128)

        pI2, pI3, pI4 = np.squeeze(bat_pred[0, ...]), np.squeeze(bat_pred[1, ...]), np.squeeze(bat_pred[2, ...])
        pI5, pI6, pI7 = np.squeeze(bat_pred[3, ...]), np.squeeze(bat_pred[4, ...]), np.squeeze(bat_pred[5, ...])
        pI8, pI9, pI10 = np.squeeze(bat_pred[6,...]), np.squeeze(bat_pred[7,...]), np.squeeze(bat_pred[8,...])
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
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale5_predict.nrrd")
        writer.Execute(sitk.GetImageFromArray(pI5))
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale6_predict.nrrd")
        writer.Execute(sitk.GetImageFromArray(pI6))
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale7_predict.nrrd")
        writer.Execute(sitk.GetImageFromArray(pI7))
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale8_predict.nrrd")
        writer.Execute(sitk.GetImageFromArray(pI8))
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale9_predict.nrrd")
        writer.Execute(sitk.GetImageFromArray(pI9))
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale10_predict.nrrd")
        writer.Execute(sitk.GetImageFromArray(pI10))

        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale_mask.nrrd")
        writer.Execute(sitk.GetImageFromArray(np.squeeze(test_sx[patient, ...])))
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale_mask2.nrrd")
        writer.Execute(sitk.GetImageFromArray(S2))
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale_mask3.nrrd")
        writer.Execute(sitk.GetImageFromArray(S3))
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale_mask4.nrrd")
        writer.Execute(sitk.GetImageFromArray(S4))
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale_mask5.nrrd")
        writer.Execute(sitk.GetImageFromArray(S5))
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale_mask6.nrrd")
        writer.Execute(sitk.GetImageFromArray(S6))
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale_mask7.nrrd")
        writer.Execute(sitk.GetImageFromArray(S7))
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale_mask8.nrrd")
        writer.Execute(sitk.GetImageFromArray(S8))
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale_mask9.nrrd")
        writer.Execute(sitk.GetImageFromArray(S9))
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "inhale_mask10.nrrd")
        writer.Execute(sitk.GetImageFromArray(S10))

        # Permute DVF #
        def dvf_(d):
            x = d[0,...]
            x = np.reshape(x, [1,128,128,128])
            y = d[1,...]
            y = np.reshape(y, [1,128,128,128])
            z = d[2,...]
            z = np.reshape(z, [1,128,128,128])
            out = np.concatenate([z,y,x],0)
            return out

        DVF2, DVF3, DVF4, DVF5 = dvf_(DVF[:,0,...]), dvf_(DVF[:,1,...]), dvf_(DVF[:,2,...]), dvf_(DVF[:,3,...])
        DVF6, DVF7, DVF8, DVF9, DVF10 = dvf_(DVF[:,4,...]), dvf_(DVF[:,5,...]), dvf_(DVF[:,6,...]), dvf_(DVF[:,7,...]),dvf_(DVF[:,8,...])


        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "DVF2.nrrd")
        writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF2), [1,2,3,0]))) # 3 1 2
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "DVF3.nrrd")
        writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF3), [1,2,3,0]))) # 3 1 2
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "DVF4.nrrd")
        writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF4), [1,2,3,0]))) # 3 1 2
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "DVF5.nrrd")
        writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF5), [1,2,3,0]))) # 3 1 2
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "DVF6.nrrd")
        writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF6), [1,2,3,0]))) # 3 1 2
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "DVF7.nrrd")
        writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF7), [1,2,3,0]))) # 3 1 2
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "DVF8.nrrd")
        writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF8), [1,2,3,0]))) # 3 1 2
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "DVF9.nrrd")
        writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF9), [1,2,3,0]))) # 3 1 2
        writer.SetFileName(savepath + "/" + "%3.3d" % patient + "/" + "DVF10.nrrd")
        writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF10), [1,2,3,0]))) # 3 1 2

if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()