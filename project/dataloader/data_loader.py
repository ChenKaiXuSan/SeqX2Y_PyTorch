'''
File: data_loader.py
Project: dataloader
Created Date: 2023-08-11 03:43:16
Author: chenkaixu
-----
Comment:
The CTDataset class to prepare the dataset for train and val.
Use a 4D CT dataset, and us SimpleITK to laod the Dicom medical image.

Have a good code time!
-----
Last Modified: Wednesday January 17th 2024 9:50:54 am
Modified By: the developer formerly known as Hao Ouyang at <ouyanghaomail@gmail.com>
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------
2023-11-20 Chen refactor the CTDataset fucntion, now it can load multi-patient data.
2023-11-20 Chen add the CT_normalize class, for normalize the CT image.
'''

import os, sys
from pathlib import Path

import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
    Lambda
)

from torchvision.transforms.functional import resize

from typing import Any, Callable, Dict, Optional, Type, Union
from pytorch_lightning import LightningDataModule

from ct_dataset import CTDataset

class CT_normalize(torch.nn.Module):
    """ CT normalize function for the CT image.
    This function to normalize the pixel value to -1~1.

    Args:
        torch (_type_): _description_
    """    

    def __init__(self, img_size = 128, x1 = 90, y1 = 80, x2 = 410, y2 = 360, *args, **kwargs) -> None:
        """_summary_

        Args:
            x1 (int, optional): _description_. Defaults to 90.
            y1 (int, optional): _description_. Defaults to 80.
            x2 (int, optional): _description_. Defaults to 410.
            y2 (int, optional): _description_. Defaults to 360.
        """        
        super().__init__(*args, **kwargs)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        # the parms for init.
        # 定义感兴趣区域的坐标范围（左上角和右下角的像素坐标）
        x1, y1 = 90, 80  # 左上角坐标
        x2, y2 = 410, 360  # 右下角坐标

        self.img_size = img_size

    def forward(self, image):
        """_summary_

        Args:
            image (_type_): _description_

        Returns:
            _type_: _description_
        """        

        # todo the logic for the normalize .
        # return normalized img.
        # dicom_image = sitk.ReadImage(image)
        # dicom_array = sitk.GetArrayFromImage(image)
        
        # -1~1 normalization
        max_value = image.max()
        min_value = image.min()
        normalized_img = 2 * ((image - min_value) / (max_value - min_value)) - 1 

        # normd_cropd_img = normalized_img[:, self.y1:self.y2, self.x1:self.x2]
        # cropd_img = image[:, self.y1:self.y2, self.x1:self.x2]

        # half_img_size = self.img_size // 2 
        center_height = image.shape[1] // 2 #！undo normalized (handle croped)
        center_width = image.shape[2] // 2
        # center_loc = normalized_img.shape[1] // 2 # do normalized (handle croped)
        bias = 180

        # croped_img = crop(normalized_img, top=center_loc-bias, left=center_loc-bias, height=bias*2, width=bias*2)
        # croped_img = crop(image, top=center_loc-bias, left=center_loc-bias, height=bias*2, width=bias*2)
        # croped_img = image[:, center_loc-180:center_loc+130, center_loc-155:center_loc+155] #！org undo normalized (handle croped)
        # croped_img = image[:, center_loc-210:center_loc+150, center_loc-175:center_loc+175] #！org 2 undo normalized (handle croped)
        croped_img = image[:, center_height-190:center_height+130, center_width-160:center_width+160] #! org 3 This is the best
        # cropped_img = image[:, top:bottom, left:right]
        # croped_img = normalized_img[:, center_loc-180:center_loc+130, center_loc-155:center_loc+155] # do normalized (handle croped)

        final_img = resize(croped_img, size=[self.img_size, self.img_size])

        return final_img

class CTDataModule(LightningDataModule):
    """
    CTDataModule, used for prepare the train/val/test dataloader.
    inherit from the LightningDataMoudle, 
    """    

    def __init__(self, train: Dict, data: Dict):
        super().__init__()

        self._TRAIN_PATH = data.data_path
        self._VAL_PATH = data.val_data_path
        self._NUM_WORKERS = data.num_workers
        self._IMG_SIZE = data.img_size
        self._BATCH_SIZE = train.batch_size
        self.vol = train.vol

        # 2D time series path
        self._TIME_SERIES_DATA_PATH = data.data_path2D
        self._VAL_TIME_SERIES_DATA_PATH = data.val_data_path2D

        # 4D CT transform
        self.train_transform = Compose(
            [
                # RandomHorizontalFlip(p=0.5),
                # CT normalize method, for every CT image normalize to 0-1 pixel value.
                CT_normalize(self._IMG_SIZE),
            ]
        )

        self.val_transform = Compose(
            [
                # ToTensor(),
                # Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                # CenterCrop([150, 150])
                CT_normalize(self._IMG_SIZE),
            ]
        )

        # 2D time series transform
        self.train_transform_time_series = Compose(
            [
                ToTensor(),
                Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                Normalize((0.45), (0.225)),
                lambda x: x/255.0,
            ]
        )

        self.val_transform_time_series = Compose(
            [
                ToTensor(),
                Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                Normalize((0.45), (0.225)),
                lambda x: x/255.0,
            ]
        )

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        '''
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        '''

        if stage in ("fit", None):
            self.train_dataset = CTDataset(
                data_path=self._TRAIN_PATH,
                data_path2D=self._TIME_SERIES_DATA_PATH,                
                ct_transform=self.train_transform,
                time_series_transform=self.train_transform_time_series,
                vol=self.vol,
            )
            
        # BUG: dataset leak.
        # ! now have dataset leak.
        if stage in ("fit", "validate", None):
            self.val_dataset = CTDataset(
                data_path=self._VAL_PATH,
                data_path2D=self._VAL_TIME_SERIES_DATA_PATH,
                ct_transform=self.val_transform,
                time_series_transform=self.val_transform_time_series,
                vol=self.vol,
            )

    def train_dataloader(self) -> DataLoader:
        '''
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        '''

        return DataLoader(
            self.train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        '''
        create the val dataloader from the list of val dataset.

        sert parameters for DataLoader prepare.        
        '''

        return DataLoader(
            self.val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )