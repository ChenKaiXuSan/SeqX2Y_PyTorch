# %%
import csv, logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorch_lightning import LightningModule
from torchmetrics import classification

from models.seq2seq_4DCT_voxelmorph import EncoderDecoderConvLSTM

# %%
class PredictLightningModule(LightningModule):

    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.img_size = hparams.data.img_size
        self.lr = hparams.optimizer.lr
        self.seq = hparams.train.seq

        self.model = EncoderDecoderConvLSTM(
            nf=96, in_chan=1, size1=128, size2=128, size3=128)

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

    def training_step(self, batch: torch.Tensor, batch_idx:int):
        '''
        train steop when trainer.fit called

        Args:
            batch (torch.Tensor): b, f, h, w
            batch_idx (int):batch index.

        Returns: None
        '''
        b, f, h, w = batch.size()

        rpm = int(np.random.randint(0, 20, 1))
        logging.info("Patient index: %s, RPM index: %s" % (batch_idx, rpm))

        RPM = np.array(self.data)
        RPM = np.float32(RPM)
        test_RPM = RPM

        # load rpm
        test_rpm_ = test_RPM[rpm,:]
        test_x_rpm = test_RPM[rpm,:1]
        test_x_rpm = np.expand_dims(test_x_rpm,0)
        test_y_rpm = test_RPM[rpm,0:]
        test_y_rpm = np.expand_dims(test_y_rpm,0)

        # invol = torch.Tensor(test_x_)
        # invol = invol.permute(0, 1, 5, 2, 3, 4)
        # invol = invol.to(device)
        invol = batch.unsqueeze(dim=0).unsqueeze(dim=0)

        test_x_rpm_tensor = torch.Tensor(test_x_rpm)
        test_y_rpm_tensor = torch.Tensor(test_y_rpm)
        test_x_rpm_tensor.cuda()
        test_y_rpm_tensor.cuda()

        # pred the video frames
        # invol: 1, 1, 1, 128, 128, 128
        # rpm_x: 1, 1
        # rpm_y: 1, 9
        bat_pred, DVF = self.model(invol, rpm_x=test_x_rpm_tensor, rpm_y=test_y_rpm_tensor, future_seq=self.seq)  # [1,2,3,176,176]

        # calc loss 
        phase_mse_loss_list = []
        phase_smooth_l1_loss_list = []

        for phase in range(self.seq):
            phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch.expand_as(bat_pred[:,:,phase,...])))
            phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch.expand_as(DVF[:,:,phase,...])))
        
        train_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))
        self.log('train_loss', train_loss)
        logging.info('train_loss: %d' % train_loss)

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        '''
        val step when trainer.fit called.

        Args:
            batch (torch.Tensor): b, f, h, w
            batch_idx (int): batch index, or patient index

        Returns: None
        '''
        b, f, h, w = batch.size()

        rpm = int(np.random.randint(0, 20, 1))
        logging.info("Patient index: %s, RPM index: %s" % (batch_idx, rpm))

        RPM = np.array(self.data)
        RPM = np.float32(RPM)
        test_RPM = RPM

        # load rpm
        test_rpm_ = test_RPM[rpm,:]
        test_x_rpm = test_RPM[rpm,:1]
        test_x_rpm = np.expand_dims(test_x_rpm,0)
        test_y_rpm = test_RPM[rpm,0:]
        test_y_rpm = np.expand_dims(test_y_rpm,0)

        # invol = torch.Tensor(test_x_)
        # invol = invol.permute(0, 1, 5, 2, 3, 4)
        # invol = invol.to(device)
        invol = batch.unsqueeze(dim=0).unsqueeze(dim=0)

        test_x_rpm_tensor = torch.Tensor(test_x_rpm)
        test_y_rpm_tensor = torch.Tensor(test_y_rpm)
        test_x_rpm_tensor.cuda()
        test_y_rpm_tensor.cuda()

        # pred the video frames
        with torch.no_grad():
            # invol: 1, 1, 1, 128, 128, 128
            # rpm_x: 1, 1
            # rpm_y: 1, 9
            bat_pred, DVF = self.model(invol, rpm_x=test_x_rpm_tensor, rpm_y=test_y_rpm_tensor, future_seq=self.seq)  # [1,2,3,176,176]

        # calc loss 
        phase_mse_loss_list = []
        phase_smooth_l1_loss_list = []

        for phase in range(self.seq):
            phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch.expand_as(bat_pred[:,:,phase,...])))
            phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch.expand_as(DVF[:,:,phase,...])))
        
        val_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))

        self.log('val_loss', val_loss)
        logging.info('val_loss: %d' % val_loss)

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
        # return torch.optim.SGD(self.parameters(), lr=self.lr)

    def _get_name(self):
        return self.model_type
