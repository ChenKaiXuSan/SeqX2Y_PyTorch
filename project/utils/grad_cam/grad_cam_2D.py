import numpy as np
import torch

from typing import Tuple, List
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from project.utils.grad_cam.activations_and_gradients_2d import ActivationsAndGradients2D
from project.loss_analyst import calculate_val_loss
from pytorch_grad_cam.utils.image import scale_cam_image


class GradCAM_2D(BaseCAM):
    """
    GradCAM implementation for 2D images.
    The source code copy from pytorch-grad-cam.

    Args:
        BaseCAM_2D (_type_): _description_
    """

    def __init__(self, model, target_layers, reshape_transform=None):
        super(GradCAM_2D, self).__init__(model, target_layers, reshape_transform)
        self.activations_and_grads = ActivationsAndGradients2D(model, target_layers, reshape_transform)

    def get_cam_weights(
        self, input_tensor, target_layer, target_category, activations, grads
    ):
        # 2D image
        if len(grads.shape) == 4:
            return np.mean(grads, axis=(2, 3))

        # 3D image
        elif len(grads.shape) == 5:
            return np.mean(grads, axis=(2, 3, 4))

        else:
            raise ValueError(
                "Invalid grads shape."
                "Shape of grads should be 4 (2D image) or 5 (3D image)."
            )

    def compute_cam_per_layer(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool
    ) -> np.ndarray:
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer
    
    def forward(
        self,
        input_tensor: dict[str, torch.Tensor],
        targets: dict[str, torch.nn.Module],
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        
        invol = input_tensor["invol"].requires_grad_(True)
        time_series_img = input_tensor["time_series_img"]
        future_seq = input_tensor["future_seq"]
        
        bat_pred = targets["bat_pred"].requires_grad_(True)
        DVF = targets["DVF"].requires_grad_(True)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [
                ClassifierOutputTarget(category) for category in target_categories
            ]

        # if self.uses_gradients:
        # FIXME: the loss need grad_fn
        #     self.model.zero_grad()
        #     loss, *_ = calculate_val_loss(bat_pred, DVF, invol, future_seq)
        #     # loss = sum([target(output) for target, output in zip(targets, outputs)])
        #     loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def __call__(
        self,
        input_tensor: dict[str, torch.Tensor],
        targets: dict[str, torch.nn.Module],
        aug_smooth: bool = False,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        """
        Modify forward to support input_tensor as a List of Tensors.
        """

        if aug_smooth is True:
            # FIXME: not implemented with list
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth
            )

        # TODO: 为了在里面计算loss，需要invol，time_series_img，future_seq，bat_pred, DVF
        return self.forward(input_tensor, targets, eigen_smooth)
