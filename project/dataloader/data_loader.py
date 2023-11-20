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
Last Modified: 2023-09-25 23:05:09
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------
2023-11-20 Chen refactor the CTDataset fucntion, now it can load multi-patient data.
2023-11-20 Chen add the CT_normalize class, for normalize the CT image.
'''

import os, sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    Resize,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
    CenterCrop,
)

from typing import Any, Callable, Dict, Optional, Type, Union
from pytorch_lightning import LightningDataModule

import SimpleITK as sitk

class CT_normalize(torch.nn.Module):
    """ CT

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

        max_value = image.max()
        min_value = image.min()
        normalized_img = (image - min_value) / (max_value - min_value)

        # normd_cropd_img = normalized_img[:, self.y1:self.y2, self.x1:self.x2]
        # cropd_img = image[:, self.y1:self.y2, self.x1:self.x2]

        half_img_size = self.img_size // 2
        center_loc = image.shape[1] // 2

        cropd_img = normalized_img[:, center_loc-half_img_size:center_loc+half_img_size, center_loc-half_img_size:center_loc+half_img_size]

        # cropd_img = image[:,30:256,100:350]

        return cropd_img
        # return normalized_img

class CTDataset(Dataset):
    def __init__(self, data_path, transform=None, vol=128):
        self.data_path = Path(data_path)
        # self.targets = targets
        self.transform = transform
        self.vol = vol
        self.all_patient_Dict = self.load_person() # Dict{number, list[Path]}

    def load_person(self,):

        patient_Dict = {}

        for i, patient in enumerate(sorted(self.data_path.iterdir())):
            # * get one patient 
            one_patient_breath_path = os.listdir(
                self.data_path / patient)

            patient_Dict[i] = self.prepare_file(self.data_path/patient, one_patient_breath_path)

        
        return patient_Dict

    def prepare_file(self, pre_path: Path, one_patient: list):

        one_patient_breath_path_List  = []

        for breath in sorted(one_patient):

            curr_path = pre_path / breath

            # here prepare the one patient all breath path.
            one_breath_full_path_List = sorted(list(iter(curr_path.iterdir())))
            one_patient_breath_path_List.append(one_breath_full_path_List)

        return one_patient_breath_path_List

    def __len__(self):
        person_number = len(self.all_patient_Dict.keys())
        breath_number = len(self.all_patient_Dict[0])
        one_breath_number = len(self.all_patient_Dict[0][0])

        return person_number

    def __getitem__(self, idx):
        """
        __getitem__, get the patient data from the patient_Dict.
        Here we need load all of the patient data, and return a 4D tensor.
        Shape like, b, c, seq, vol, h, w

        Args:
            idx (_type_): not use here.

        Returns:
            torch.Tensor: the patient data, shape like, b, c, seq, vol, h, w
        """        

        one_patient_full_vol = []

        for breath_path in self.all_patient_Dict[idx]: # one patient path 
            

            one_breath_img = []

            for img_path in breath_path:
                image = sitk.ReadImage(img_path)
                image_array = sitk.GetArrayFromImage(image)
                if self.transform:
                    # c, h, w
                    image_array = self.transform(torch.from_numpy(image_array).to(torch.float32))
                one_breath_img.append(image_array)
                
                # FIXME: this is that need 128 for one patient, for sptail transformer, in paper.
                # ! or should unifrom extract 128 from all vol, not from start to index.
                if len(one_breath_img) == self.vol:
                    break;
            # c, h, w
            one_patient_full_vol.append(torch.stack(one_breath_img, dim=1)) # c, v, h, w

        return torch.stack(one_patient_full_vol, dim=0) # seq, c, v, h, w


class CTDataModule(LightningDataModule):
    """
    CTDataModule, used for prepare the train/val/test dataloader.
    inherit from the LightningDataMoudle, 
    """    

    def __init__(self, train, data):
        super().__init__()

        self._TRAIN_PATH = data.data_path
        self._NUM_WORKERS = data.num_workers
        self._IMG_SIZE = data.img_size
        self._BATCH_SIZE = train.batch_size
        self.vol = train.vol

        self.train_transform = Compose(
            [
                # ToTensor(),
                # Normalize((0.45), (0.225)),
                # RandomCrop(self._IMG_SIZE),
                # Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                RandomHorizontalFlip(p=0.5),
                # CenterCrop([150, 150])
                # CT normalize method, for every CT image normalize to 0-1 pixel value.
                CT_normalize(self._IMG_SIZE),
            ]
        )

        self.val_transform = Compose(
            [
                # ToTensor(),
                Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                # CenterCrop([150, 150])
                CT_normalize(self._IMG_SIZE),
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

        # if stage == "fit" or stage == None:
        if stage in ("fit", None):
            self.train_dataset = CTDataset(
                data_path=self._TRAIN_PATH,
                transform=self.train_transform,
                vol=self.vol,
            )

        if stage in ("fit", "validate", None):
            self.val_dataset = CTDataset(
                data_path=self._TRAIN_PATH,
                transform=self.val_transform,
                vol=self.vol,
            )

        # if stage in ("predict", "test", None):
        #     self.test_pred_dataset = WalkDataset(
        #         data_path=os.path.join(data_path, "val"),
        #         clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
        #         transform=transform
        #     )

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

    def test_dataloader(self) -> DataLoader:
        '''
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and 
        normalizes the video before applying the scale, crop and flip augmentations.
        '''
        return DataLoader(
            self.val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )