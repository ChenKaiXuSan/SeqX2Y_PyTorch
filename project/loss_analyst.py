#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/SeqX2Y_PyTorch/project/eval_function.py
Project: /workspace/SeqX2Y_PyTorch/project
Created Date: Thursday January 11th 2024
Author: Hao Ouyang
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday January 17th 2024 9:50:54 am
Modified By: the developer formerly known as Hao Ouyang at <ouyanghaomail@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

11-01-2024	Hao Ouyang	
'''
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
import matplotlib.pyplot as plt
# use [ctrl + shift + 2] to insert function description
# use >Header Change Log Insert to insert the head log

# Calculate smoothness loss for DVF
def calculate_smoothness_loss(dvf):
    """Calculate smoothness loss for DVF(in paper use smoothness loss, but result is not good)

    Args:
        dvf 

    Returns:
        smoothness_loss.mean(dim=[1, 2, 3, 4])
    """    
    # Assuming dvf is a 6D tensor: batch x seq x channels x depth x height x width
    dvf_grad_x = torch.gradient(dvf, dim=4, spacing=(1,))[0]
    dvf_grad_y = torch.gradient(dvf, dim=3, spacing=(1,))[0]
    dvf_grad_z = torch.gradient(dvf, dim=2, spacing=(1,))[0]
    # Summing the squares of the gradients
    smoothness_loss = dvf_grad_x.pow(2) + dvf_grad_y.pow(2) + dvf_grad_z.pow(2)
    # smoothness_loss = dvf_grad_x**2 + dvf_grad_y**2 + dvf_grad_z**2
    # Taking the mean over all dimensions except the batch
    return smoothness_loss.mean(dim=[1, 2, 3, 4])

# def calculate_ssim(self, x, y):
#     _, _, depth, _, _ = x.shape
#     ssim_scores = []

#     # 确保x和y转换为numpy数组，因为skimage的SSIM函数需要numpy数组
#     x_np = x[0].cpu().detach().numpy()
#     y_np = y[0].cpu().detach().numpy()

#     # 遍历每个深度切片
#     for d in range(depth):
#         # 计算每个深度切片的SSIM
#         ssim_value = SSIM(x_np[:, d, ...], y_np[:, d, ...])
#         ssim_scores.append(ssim_value)

#     # 返回所有深度切片的平均SSIM
#     return sum(ssim_scores) / len(ssim_scores)

# # Calculate NCC values: All dimensions together
# def normalized_cross_correlation(self, x, y):
#     mean_x = torch.mean(x) # x torch.Size([1, 1, 118, 128, 128])
#     mean_y = torch.mean(y) # y torch.Size([1, 1, 118, 128, 128])
#     x_normalized = x - mean_x
#     y_normalized = y - mean_y
#     ncc = torch.sum(x_normalized * y_normalized) / (torch.sqrt(torch.sum(x_normalized ** 2)) * torch.sqrt(torch.sum(y_normalized ** 2)))
#     return ncc

# Calculate NCC values: Depth dimension only
def normalized_cross_correlation(x, y):
    """Calculate NCC values: Depth dimension only

    Args:
        x 
        y 

    Returns:
        sum(ncc_scores) / len(ncc_scores): Average NCC
    """    
    # batch_size = x.shape[2]
    batch_size, channels, depth, _, _ = x.shape
    ncc_scores = []

    # Traverse each sample in depth
    for d in range(depth):
        # Average
        mean_x = torch.mean(x[:,:,d,...])
        mean_y = torch.mean(y[:,:,d,...])
        # normalized
        x_normalized = x[:,:,d,...] - mean_x
        y_normalized = y[:,:,d,...] - mean_y
        # NCC
        ncc = torch.sum(x_normalized * y_normalized) / (torch.sqrt(torch.sum(x_normalized ** 2)) * torch.sqrt(torch.sum(y_normalized ** 2)))
        ncc_scores.append(ncc.item())
    # Average NCC
    return sum(ncc_scores) / len(ncc_scores)

# # Calculate Dice values: All dimensions together
# def dice_coefficient(self, pred, target):
#     smooth = 1.0  # Used to prevent division by zero
#     # Binarize the prediction and target, and the threshold is usually set at 0.5
#     pred = (pred > 0).float()
#     target = (target > 0).float()
#     # intersection = (pred * target).sum()
#     intersection = torch.sum(pred*target, dim=[2, 3, 4])
#     # dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
#     dice = (2. * intersection + smooth) / (torch.sum(pred, dim=[2, 3, 4]) + torch.sum(target, dim=[2, 3, 4]) + smooth)
#     return dice

# Calculate Dice values: Depth only
def dice_coefficient(x, y):
    """Calculate Dice values: Depth only

    Args:
        x 
        y 
    Returns:
        sum(dice_scores) / len(dice_scores): Average Dice
    """    
    _, _, depth, _, _ = x.shape
    dice_scores = []

    # Convert x and y values from [-1, 1] to [0, 1]
    x = (x > 0).float()
    y = (y > 0).float()

    # Traverse each sample in depth
    for d in range(depth):
        # intersection
        intersection = (x[:, :, d, ...] * y[:, :, d, ...]).sum()
        # Dice
        dice = (2. * intersection) / (x[:, :, d, ...].sum() + y[:, :, d, ...].sum())
        dice_scores.append(dice.item())
    # Average Dice
    return sum(dice_scores) / len(dice_scores)

# Calculate TRE values
def calculate_tre(points_pred, points_true):
    """
    :param points_pred: 预测的点的位置，形状为 [N, 3]，其中 N 是点的数量
    :param points_true: 真实的点的位置，形状为 [N, 3]
    :return: TRE 值
    """
    # 确保预测点和真实点的数量相同
    assert points_pred.shape == points_true.shape, "预测点和真实点的数量和/或维度不匹配"
    # 计算点对之间的欧氏距离
    tre = torch.sqrt(torch.sum((points_pred - points_true) ** 2, dim=1))
    return torch.mean(tre)





# Caluate training loss
def calculate_train_loss(bat_pred, DVF, ct_data, seq):
    """Caluate training loss 
    Please view the internal details

    Args:
        bat_pred (tensor): _description_
        DVF (tensor): _description_
        ct_data (tensor): _description_
        seq (int): _description_

    Returns:
        train_loss: tensor
    """    
    # Calc Loss 
    phase_mse_loss_list = []
    phase_smooth_l1_loss_list = []

    # chen orign 
    # for phase in range(self.seq):
    for phase in range(seq): #In4 Out3 use seq-1
        phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], ct_data[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))   # DVF torch.Size([1, 3, 3, 70, 120, 140])
        phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], ct_data[:, phase, ...].expand_as(DVF[:,:,phase,...]))) # DVF[:,:,phase,...] torch.Size([1, 3, 70, 120, 140])          
        #!FIXME Metrics Test But Erro ValueError: Expected both prediction and target to be 1D or 2D tensors, but received tensors with dimension torch.Size([1, 1, 118, 128, 128])
        # mse_value = self.mse(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...]))
        # mae_value = self.mae(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...]))
        #r2_value = self.r2_score(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...]))                
    train_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))

    # # Origin Loss Function
    # for phase in range(self.seq):
    #     # MSE loss
    #     phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))   # bat_pred[:,:,phase,...].shape => torch.Size([1, 1, 118, 128, 128])
    #     # smooth l1 loss
    #     phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase, ...].expand_as(DVF[:,:,phase,...]))) # DVF[:,:,phase,...].shape => torch.Size([1, 3, 118, 128, 128])                     
    # # sum two loss
    # train_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))

    # wrong 1 3 5 7 --> 2 4 6 8 
    # for phase in range(self.seq):
    # for phase in range(self.seq-4):
    #     # +1 表示让预测生成的肺与后一个肺做loss
    #     phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:, phase*2+1, ...].expand_as(bat_pred[:,:,phase,...]))) # bat_pred(1,1,3,128,128,128), batch torch.Size([1, 4, 70, 120, 140])
    #     phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase*2+1, ...].expand_as(DVF[:,:,phase,...]))) # DVF[:,:,phase,...] torch.Size([1, 3, 70, 120, 140])              
    # train_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))

    # # Right 1 3 5  --> 2 4 6 
    # for phase in range(0, batch.shape[1], 2): 
    #     # +1 loss was made between the predicted lung and the lung at t+1 time
    #     phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase//2,...], batch[:, phase+1, ...].expand_as(bat_pred[:,:,phase//2,...])))     # bat_pred(1,1,3,128,128,128), batch torch.Size([1, 4, 70, 120, 140])
    #     phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase//2,...], batch[:, phase+1, ...].expand_as(DVF[:,:,phase//2,...])))   # DVF[:,:,phase,...] torch.Size([1, 3, 70, 120, 140])              
    # train_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))


    # ouyangV1 add spatial transform
    # Transform = Warp(size1=128, size2=128, size3=128).cuda() # spatial transform 
    # for phase in range(self.seq):
    #     T = Transform(bat_pred[:,:,phase,...], batch[:, 0, ...].expand_as(bat_pred[:,:,0,...]))
    #     phase_mse_loss_list.append(F.mse_loss(T, batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))
    #     phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase, ...].expand_as(DVF[:,:,phase,...])))
    # train_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))
    # self.log('train_loss', train_loss)
    # logging.info('train_loss: %d' % train_loss)

    # ouyangV2 add gradient
    # for phase in range(self.seq):
    #     phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))
    #     # phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase, ...].expand_as(DVF[:,:,phase,...])))
    #     input_tensor = batch[:, phase, ...].expand_as(DVF[:,:,phase,...])
    #     input_tensor.requires_grad = True
    #     gradient_phi_t = torch.autograd.grad(outputs=DVF[:,:,phase,...], inputs=input_tensor, grad_outputs=torch.ones_like(DVF[:,:,phase,...]), create_graph=True)[0]
    #     part2_loss = torch.sum(gradient_phi_t.pow(2))
    # train_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(part2_loss, dim=0))

    # ouyangV3 smoothness loss
    # for phase in range(self.seq):
    #     phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))
    #     phase_smooth_l1_loss_list.append(self.calculate_smoothness_loss(DVF[:,:,phase,...]))
    # train_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))
    # self.log('train_loss', train_loss)
    # logging.info('train_loss: %d' % train_loss)
    
    return train_loss

# Caluate validation loss
def calculate_val_loss(bat_pred, DVF, ct_data, seq):
    """Caluate validation loss 
    Please view the internal details

    Args:
        bat_pred (_type_): _description_
        DVF (_type_): _description_
        ct_data (_type_): _description_
        seq (_type_): _description_

    Returns:
        val_loss, ssim_values, ncc_values, dice_values
    """    
    # calc loss 
    phase_mse_loss_list = []
    phase_smooth_l1_loss_list = []
    # SSIM
    ssim_values = []
    ssim = SSIM().to(device=1) # data_range = 2
    # NCC
    ncc_values = []
    # DICE
    dice_values = []

    # Orign Chen+SSIM+NCC+DICE
    for phase in range(seq):
        phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], ct_data[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))   # DVF torch.Size([1, 3, 3, 70, 120, 140])
        phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], ct_data[:, phase, ...].expand_as(DVF[:,:,phase,...])))

        # ssim
        ssim_value = ssim(bat_pred[:,:,phase,...], ct_data[:, phase, ...].expand_as(bat_pred[:, :, phase,...]))
        # ssim_values.append(ssim_value.item())
        ssim_values.append(ssim_value.item())
        # ncc
        ncc_value = normalized_cross_correlation(bat_pred[:,:,phase,...], ct_data[:,phase,...].expand_as(bat_pred[:,:,phase,...]))
        # ncc_values.append(ncc_value.item())
        ncc_values.append(ncc_value)
        # dice
        dice_value = dice_coefficient(bat_pred[:,:,phase,...], ct_data[:,phase,...].expand_as(bat_pred[:,:,phase,...]))
        dice_values.append(dice_value)

    val_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))

    # # chen ORIGIN
    # for phase in range(self.seq):
    # # for phase in range(self.seq-1):
    #     phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:,phase,...].expand_as(bat_pred[:,:,phase,...])))  # DVF torch.Size([1, 3, 3, 70, 120, 140]), batch torch.Size([1, 4, 70, 120, 140])
    #     phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase, ...].expand_as(DVF[:,:,phase,...]))) # but DVF[:,:,phase,...] torch.Size([1, 3, 70, 120, 140])
    # val_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))

    # self.log('val_loss', val_loss)
    # logging.info('val_loss: %.4f' % val_loss)

    # ouyangV1
    # Transform = Warp(size1=128, size2=128, size3=128).cuda() # spatial transform
    # for phase in range(self.seq):
    #     T = Transform(bat_pred[:,:,phase,...], batch[:, 0, ...].expand_as(bat_pred[:,:,0,...]))
    #     phase_mse_loss_list.append(F.mse_loss(T, batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))
    #     phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase, ...].expand_as(DVF[:,:,phase,...])))
    # val_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))
    
    # ||∇ϕt||^2
    # for phase in range(self.seq):
    #     phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))
    #     # phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch[:, phase, ...].expand_as(DVF[:,:,phase,...])))
    #     DDD=DVF[:,:,phase,...]
    #     DDD.requires_grad = True
    #     gradient_phi_t = torch.autograd.grad(DDD.sum(), DDD, create_graph=True)[0]
    #     part2_loss = torch.sum(gradient_phi_t.pow(2))
    # val_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(part2_loss, dim=0))
    
    # ouyangV3 smoothness loss
    # for phase in range(self.seq):
    #     phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch[:, phase, ...].expand_as(bat_pred[:,:,phase,...])))
    #     phase_smooth_l1_loss_list.append(self.calculate_smoothness_loss(DVF[:,:,phase,...]))
    # val_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))
    # self.log('val_loss', val_loss)
    # logging.info('val_loss: %d' % val_loss)

    return val_loss, ssim_values, ncc_values, dice_values


def draw_image(average_ssim, average_ncc, average_dice):
    """To draw the SSIM, NCC, Dice's image

    Args:
        average_ssim (_type_): _description_
        average_ncc (_type_): _description_
        average_dice (_type_): _description_
    """    
    # Draw the image
    metrics = ['SSIM', 'NCC', 'DICE']
    # average_dice_cpu = average_dice.cpu().item()
    values = [average_ssim, average_ncc, average_dice]  # 使用 .item() 转换 PyTorch 张量为 Python 数字
    # #  STYLE 1 draw bar picture
    # plt.figure(figsize=(10, 5))
    # plt.bar(metrics, values, color=['blue', 'green', 'red'])
    # plt.title('Average Metric Values')
    # plt.xlabel('Metrics')
    # plt.ylabel('Values')
    # # show value
    # for i, v in enumerate(values):
    #     plt.text(i, v + 0.01, "{:.4f}".format(v), ha='center', va='bottom')
    # plt.show()
    # plt.savefig('/workspace/SeqX2Y_PyTorch/test/Imageresult/matplot1.png')

    # STYLE 2
    plt.style.use('ggplot')
    # 创建一个条形图
    fig, ax = plt.subplots(figsize=(10, 5))  # 可以调整大小以适应您的需求
    # 绘制条形图
    bars = ax.bar(metrics, values, color=['salmon', 'cornflowerblue', 'teal'], width=0.5, edgecolor='black', linewidth = 0)
    # 添加数值标签
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 4), va='bottom', ha='center')
    # 设置标题和标签
    ax.set_title('Average Metric Values', fontsize=16)
    ax.set_xlabel('Metrics', fontsize=14)
    ax.set_ylabel('Values', fontsize=14)
    # 设置 y 轴的限制
    ax.set_ylim(0, 1)
    # 移除顶部和右侧的边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 显示网格线
    ax.yaxis.grid(True)
    # 设置 y 轴刻度标签的大小
    ax.tick_params(axis='y', labelsize=12)
    # 显示图表
    plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
    plt.show()
    plt.savefig('/workspace/SeqX2Y_PyTorch/test/Imageresult/matplot2.png')

    # # STYLE 3 heatmap
    # # 创建一个单行的矩阵，每个指标一个值
    # heatmap_data = np.array(values).reshape(1, len(values))
    # # 使用 seaborn 创建热力图
    # plt.figure(figsize=(8, 2))
    # sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap='coolwarm', xticklabels=metrics, yticklabels=False)
    # plt.title('Metric Heatmap')
    # plt.savefig('/workspace/SeqX2Y_PyTorch/test/Imageresult/matplot3.png')
    # plt.show()