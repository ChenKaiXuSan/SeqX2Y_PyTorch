import numpy as np
import torch
import torch.nn.functional as F

from typing import Tuple, List
from project.utils.base_cam_2D import BaseCAM_2D
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class GradCAM_2D(BaseCAM_2D):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            GradCAM_2D,
            self).__init__(
            model,
            target_layers,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        # 2D image
        if len(grads.shape) == 4:
            return np.mean(grads, axis=(2, 3))
        
        # 3D image
        elif len(grads.shape) == 5:
            return np.mean(grads, axis=(2, 3, 4))
        
        else:
            raise ValueError("Invalid grads shape." 
                             "Shape of grads should be 4 (2D image) or 5 (3D image).")
     
        
    def __call__(self, input_tensor: List[torch.Tensor], targets: List[torch.nn.Module], future_seq: int = 3, eigen_smooth: bool = False) -> np.ndarray:
        """
        Modify forward to support input_tensor as a List of Tensors.
        """
        # 如果 input_tensor 是 List，逐个处理每个 Tensor
        if isinstance(input_tensor, list):
            # 存储每个 Tensor 计算出的 Grad-CAM 结果
            cam_results = []
            
            # 从 input_tensors 中解包 x 和 batch_2D
            invol, batch_2D = input_tensor
            invol = invol.to(self.device)
            batch_2D = batch_2D.to(self.device)
            if self.compute_input_gradient:
                invol = torch.autograd.Variable(invol, requires_grad=True)

            # 获取模型的输出，同时传递 batch_2D 和 future_seq
            self.outputs = outputs = self.activations_and_grads(invol, batch_2D, future_seq) # self.outputs len = 5

            # 处理目标类别
            if targets is None:
                target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
                targets = [ClassifierOutputTarget(category) for category in target_categories]

            # 计算损失并反向传播
            if self.uses_gradients:
                self.model.zero_grad()
                loss = sum([target(output) for target, output in zip(targets, outputs)])
                loss.backward(retain_graph=True)

            # 计算 Grad-CAM
            cam_per_layer = self.compute_cam_per_layer(invol, targets, eigen_smooth)
            cam_results.append(self.aggregate_multi_layers(cam_per_layer))
            
            # 返回所有计算出的 Grad-CAM 结果
            return np.stack(cam_results)
        
        # 如果 input_tensor 不是 List，则按原方式处理单个 Tensor
        else:
            input_tensor = input_tensor.to(self.device)

            if self.compute_input_gradient:
                input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

            # 获取模型的输出
            self.outputs = outputs = self.activations_and_grads(input_tensor)

            if targets is None:
                target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
                targets = [ClassifierOutputTarget(category) for category in target_categories]

            if self.uses_gradients:
                self.model.zero_grad()
                loss = sum([target(output) for target, output in zip(targets, outputs)])
                loss.backward(retain_graph=True)

            cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
            return self.aggregate_multi_layers(cam_per_layer)