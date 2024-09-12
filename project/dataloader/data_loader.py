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
from torchvision.utils import save_image

from typing import Any, Callable, Dict, Optional, Type, Union
from pytorch_lightning import LightningDataModule

from ct_dataset import CTDataset

# split 2D images into different part, to test model's performance
def split_image_half(img):
    """
    Split the image into left and right halves.
    """
    middle_index = img.size(2) // 2  # Assuming img is in CxHxW format
    img = img - img.min()  # Translate to positive values
    img = img / img.max()  # Scale to [0, 1]
    left_half = img[:, :, :middle_index]
    right_half = img[:, :, middle_index:]
    up_half = img[:, :middle_index, :] #不行，分上下会报错？！修改了Resize，现在不会报错了
    bottom_half = img[:, middle_index:, :]
    # return torch.cat([left_half, right_half], dim=0)  # Concatenate along the channel dimension

    bottom_half = Resize(size=[128, 128])(bottom_half)
    # save_image(left_half, '/home/ec2-user/SeqX2Y_PyTorch/test/Imageresult/Resize2D/left_half.png')
    return bottom_half 
    # return Resize(size=[128, 128])(left_half)

def split_image_quarters(img):
    """
    Split the image into four quarters: top-left, top-right, bottom-left, bottom-right.
    """
    # Assuming img is in CxHxW format
    height_middle_index = img.size(1) // 2
    width_middle_index = img.size(2) // 2
    img = img - img.min()  # Translate to positive values
    img = img / img.max()  # Scale to [0, 1]
    top_left = img[:, :height_middle_index, :width_middle_index]
    top_right = img[:, :height_middle_index, width_middle_index:]
    bottom_left = img[:, height_middle_index:, :width_middle_index]
    bottom_right = img[:, height_middle_index:, width_middle_index:]

    save_image(img, '/home/ec2-user/SeqX2Y_PyTorch/test/Imageresult/Resize2D/img.png')
    save_image(bottom_right, '/home/ec2-user/SeqX2Y_PyTorch/test/Imageresult/Resize2D/top_right.png')
    save_image(Resize(size=[128, 128])(bottom_right), '/home/ec2-user/SeqX2Y_PyTorch/test/Imageresult/Resize2D/Resized_top_right.png')   
    # Concatenate along the channel dimension
    # This results in a tensor with 4x the number of channels of the input
    # return torch.cat([top_left, top_right, bottom_left, bottom_right], dim=0)
    return Resize(size=[128, 128])(bottom_right)

def split_image_sixteenths(img, selected_index=15):
    """
    Split the image into sixteen equal parts and return the part specified by `selected_index`.
    
    Args:
    img (Tensor): Input image tensor in CxHxW format.
    selected_index (int): Index of the part to return (0 to 15).
    """
    C, H, W = img.shape  # Assuming img is in CxHxW format
    height_quarter_index = H // 4
    width_quarter_index = W // 4

    # Normalize image to [0, 1]
    img = img - img.min()  # Translate to positive values
    img = img / img.max()  # Scale to [0, 1]

    parts = []
    for i in range(4):
        for j in range(4):
            part = img[:, i * height_quarter_index:(i + 1) * height_quarter_index,
                       j * width_quarter_index:(j + 1) * width_quarter_index]
            parts.append(part)
            # Optionally save each part
            # save_image(part, f'/home/ec2-user/SeqX2Y_PyTorch/test/Imageresult/Resize2D/16/part_{i*4+j+1}.png')
            # save_image(Resize(part, size=(128, 128)), f'/home/ec2-user/SeqX2Y_PyTorch/test/Imageresult/Resize2D/16/resized_part_{i*4+j+1}.png')
    
    # save_image(parts[selected_index], '/home/ec2-user/SeqX2Y_PyTorch/test/Imageresult/Resize2D/1_16.png')
    # save_image(Resize(size=[128, 128])(parts[selected_index]), '/home/ec2-user/SeqX2Y_PyTorch/test/Imageresult/Resize2D/Resized_1_16.png')   
    
    # Return the selected part
    return Resize(size=[128, 128])(parts[selected_index])

# 归一化处理测试
def zero2one(img):
    """
    原本img直接保存显示为全黑，将其进行0-1归一后再保存显示.
    """
    img = img - img.min()  # Translate to positive values
    img = img / img.max()  # Scale to [0, 1]
    save_image(img, '/home/ec2-user/SeqX2Y_PyTorch/test/Imageresult/Resize2D/img.png')
    return img

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
        center_loc = normalized_img.shape[1] // 2 # do normalized (handle croped)
        bias = 180

        # croped_img = crop(normalized_img, top=center_loc-bias, left=center_loc-bias, height=bias*2, width=bias*2)
        # croped_img = crop(image, top=center_loc-bias, left=center_loc-bias, height=bias*2, width=bias*2)
        # croped_img = image[:, center_loc-180:center_loc+130, center_loc-155:center_loc+155] #！org undo normalized (handle croped)
        # croped_img = image[:, center_loc-210:center_loc+150, center_loc-175:center_loc+175] #！org 2 undo normalized, 02-19crossval use this
        croped_img = image[:, center_height-190:center_height+130, center_width-160:center_width+160] #! org 3 This is the best, 03-03crossval use this
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
                # split_image_half,  # Add the split image function here
                # split_image_quarters,
                split_image_sixteenths, # 1/16
                # zero2one,
            ]
        )

        self.val_transform_time_series = Compose(
            [
                ToTensor(),
                Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                Normalize((0.45), (0.225)),
                lambda x: x/255.0,
                # split_image_half,  # Add the split image function here
                # split_image_quarters,
                split_image_sixteenths, # 1/16
                # zero2one,
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