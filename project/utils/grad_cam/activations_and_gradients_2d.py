#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /home/xchen/workspace/SeqX2Y_PyTorch/project/utils/grad_cam/activations_and_gradients_2d.py
Project: /home/xchen/workspace/SeqX2Y_PyTorch/project/utils/grad_cam
Created Date: Saturday October 12th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday October 12th 2024 3:39:51 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

class ActivationsAndGradients2D(ActivationsAndGradients):

    def __init__(self, model, target_layers, reshape_transform=None):
        super(ActivationsAndGradients2D, self).__init__(model, target_layers, reshape_transform)

    def __call__(self, input_tensor):
        self.gradients = []
        self.activations = []

        invol = input_tensor["invol"]
        time_series_img = input_tensor["time_series_img"]
        future_seq = input_tensor["future_seq"]

        return self.model(invol, time_series_img, future_seq)