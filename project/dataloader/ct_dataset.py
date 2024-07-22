#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/SeqX2Y_PyTorch/project/dataloader/4DCT_dataset.py
Project: /workspace/SeqX2Y_PyTorch/project/dataloader
Created Date: Wednesday January 10th 2024
Author: Kaixu Chen
-----
Comment:
This script used for prepare the 4DCT dataset, and the corresponding time series data.
Here we set two data path into one dataset, and return a dict name sample_info_dict.
The sample_info_dict structure is:
sample_info_dict = {
    'patient_id': idx,
    '4DCT': torch.stack(one_patient_full_vol, dim=0),
    '2D_time_series': torch.stack(one_patient_time_series, dim=1) # c, t, h, w
}
The returned dict used by the train/val process, in tran.py file.

Have a good code time :)
-----
Last Modified: Monday July 22nd 2024 5:16:06 am
Modified By: the developer formerly known as Hao Ouyang at <ouyanghaomail@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
import os
import sys
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from typing import Any, Callable, Dict, Optional, Type, Union
from PIL import Image

import SimpleITK as sitk
import numpy as np


class CTDataset(Dataset):
    def __init__(self, data_path, data_path1D, ct_transform=None, time_series_transform=None, vol=128):
        """init the params for the CTDataset.

        Args:
            data_path (str): main path for the dataset.
            transform (dict, optional): the transform used for dataset. Defaults to None.
            vol (int, optional): the limited of the vol. Defaults to 128.
        """
        self.vol = vol

        # 1D coordinate time series
        self.data_path1D = Path(data_path1D)
        with open(self.data_path1D, 'r', encoding='utf-8-sig') as f:
            self.time_series = list(csv.reader(f))

        

        # 2D time series
        # self.time_image_List2D = self._load_samples()
        self.time_image_transform = time_series_transform

        # CT
        self.data_path = Path(data_path)
        self.all_patient_Dict = self.load_person()  # Dict{number, list[Path]}
        self.transform = ct_transform

    def _load_samples(self):
        """
        加载所有时间序列文件夹内的图像文件路径。
        返回:
            list: 包含所有图像文件路径的列表。
        """
        # samples = []
        # # 遍历数据集目录
        # for sequence_folder in sorted(self.data_path2D.iterdir()):
        #     if sequence_folder.is_dir():
        #         # 遍历序列文件夹内的所有图像文件
        #         sequence_images = sorted(sequence_folder.glob('*.png'))
        #         samples.append(sequence_images)
        # return samples

        # cross-val
        samples = []
        # shell读取路径转为list
        data_path2D_list = str(self.data_path2D).split(',')
        # 遍历数据集目录
        for sequence_folder in sorted(data_path2D_list):
        # for sequence_folder in sorted(self.data_path2D.iterdir()):
            # if sequence_folder.is_dir():
            
            # one patient 
            one_patient_path = Path(sequence_folder)
                # 遍历序列文件夹内的所有图像文件
            sequence_images = sorted(one_patient_path.glob('*.png'))
            samples.append(sequence_images)
        return samples
    
    def load_person(self,):
        """prepare the patient data, and return a Dict.
        Load from a main path, like: /workspace/data/POPI_dataset
        key is the patient number, value is the patient data.

        Returns:
            Dict: patient data Dict.
        """

        # patient_Dict = {}

        # for i, patient in enumerate(sorted(self.data_path.iterdir())):
        #     # * get one patient
        #     one_patient_breath_path = os.listdir(
        #         self.data_path / patient)

        # cross-val
        patient_Dict = {}
        # shell读取路径转为list
        data_path_list = str(self.data_path).split(',')
        # for i, patient in enumerate(sorted(self.data_path.iterdir())):
        for i, patient in enumerate(sorted(data_path_list)):
            # * get one patient
            one_patient_breath_path = os.listdir(
                self.data_path / patient)

            patient_Dict[i] = self.prepare_file(
                self.data_path/patient, one_patient_breath_path)

        return patient_Dict

    def prepare_file(self, pre_path: Path, one_patient: list):

        one_patient_breath_path_List = []

        for breath in sorted(one_patient):

            curr_path = pre_path / breath

            # here prepare the one patient all breath path.
            one_breath_full_path_List = sorted(list(iter(curr_path.iterdir())))
            one_patient_breath_path_List.append(one_breath_full_path_List)

        return one_patient_breath_path_List

    def __len__(self):
        """get the length of the dataset.
        person_number: the total number of patients.
        breath_number: the total number of breath for one patient.
        one_breath_number: the total number of image for one breath, in detail path.

        Returns:
            int: depends on the __getitem__ idx, here is the person_number.
        """

        person_number = len(self.all_patient_Dict.keys())
        breath_number = len(self.all_patient_Dict[0])
        one_breath_number = len(self.all_patient_Dict[0][0])
        # 2D time series
        # person_number_2D = len(self.time_image_List2D)

        # 1D coordinate time series
        person_number_1D = len(self.time_series)

        return person_number

    def save_normalize_GT_images(self, final_images, idx):
        """
        累积处理的图像并保存为CT图像格式, 注:目前不用此方法了, 直接在train.py中ct_data处进行保存更好  
        Args:
            final_images (List[torch.Tensor]): 处理后的图像列表，每个图像形状为[1, 128, 128]
            而final_images形状应为[self.vol,128,128]
        """
        # 将图像沿第0维合并
        combined_img = torch.cat(final_images, dim=0)  # combined_img尺寸为[128, 128, 128]
        # 将PyTorch Tensor转换为NumPy数组
        combined_img_np = combined_img.numpy()
        # 使用SimpleITK将NumPy数组转换为SimpleITK的Image对象
        combined_img_sitk = sitk.GetImageFromArray(combined_img_np)
        # 假设combined_img_sitk是一个SimpleITK的Image对象，它是从浮点数数据创建的，你需要先将它转换为NumPy数组进行处理
        combined_img_np = sitk.GetArrayFromImage(combined_img_sitk)
        # 将浮点数数据标准化到特定的范围，例如0到255，然后转换为整数（如uint8），这里假设combined_img_np的数据范围是-1到1
        normalized_img = ((combined_img_np - combined_img_np.min()) / (combined_img_np.max() - combined_img_np.min()) * 255).astype(np.uint8)
        # 将处理后的NumPy数组转换回SimpleITK的Image对象
        processed_img_sitk = sitk.GetImageFromArray(normalized_img)
        # 保存为CT图像
        # writer = sitk.ImageFileWriter()
        # writer.SetFileName(os.path.join("/workspace/SeqX2Y_PyTorch/test/Imageresult/GT", f"{idx}GT_normalize.dcm"))
        # writer.Execute(sitk.GetImageFromArray(final_images))
        sitk.WriteImage(processed_img_sitk, f"/workspace/SeqX2Y_PyTorch/test/Imageresult/GT/GT_normalize_{idx}.nrrd") # 或者改成.dcm也可以保存

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
        one_patient_time_series = []

        # check shape of two dataset.
        # assert len(self.all_patient_Dict[idx]) == len(
        #     self.time_series[idx]), "the shape of two dataset is not same."

        # * Step1: prepare the 4DCT data.
        for breath_path in self.all_patient_Dict[idx]:  # one patient path

            one_breath_img = []
            choose_slice_one_breath_img = []

            for img_path in breath_path:
                image = sitk.ReadImage(img_path)
                image_array = sitk.GetArrayFromImage(image)
                if self.transform:
                    # c, h, w
                    image_array = self.transform(
                        torch.from_numpy(image_array).to(torch.float32))
                one_breath_img.append(image_array)
                # choose start slice to put into the one_breath_img
                if len(one_breath_img) > 3: # 0219crossval use 20 v=118, 0303use 10 v=128
                    choose_slice_one_breath_img.append(image_array)
                # FIXME: this is that need 128 for one patient, for sptail transformer, in paper.
                # ! or should unifrom extract 128 from all vol, not from start to index.
                # if len(one_breath_img) == self.vol:
                #     break;
                if len(choose_slice_one_breath_img) == self.vol:
                    # self.save_normalize_GT_images(choose_slice_one_breath_img,idx) # save the GT images
                    break
            # c, h, w
            # one_patient_full_vol.append(torch.stack(one_breath_img, dim=1)) # c, v, h, w
            one_patient_full_vol.append(torch.stack(
                choose_slice_one_breath_img, dim=1))  # c, v, h, w

        # * Step2: prepare the 2D time series data.
        # load coordinate time series data
        part = str(breath_path[0]).split('/')[3]
        part = [int(i) for i in part]

        for p in part:
            one_patient_time_series.append(self.time_series[idx][p])
                
        # * Step3: put the 4DCT and 2D time series data into a dict.
        sample_info_dict = {
            'patient_id': idx,
            '4DCT': torch.stack(one_patient_full_vol, dim=0),
            # c, t, h, w
            '1D_time_series': one_patient_time_series
        }

        return sample_info_dict
