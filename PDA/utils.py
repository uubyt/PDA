# import torch
# import torch.nn as nn

# class Loss(nn.Module):
#     def __init__(self):
#         super(Loss, self).__init__()

#     def forward(self, output, target):
#         loss = torch.abs(output - target)  # 例子: 使用绝对差作为损失
#         return loss.mean()
import torch.nn.functional as F

def compute_loss(output, target, padding_value=0):
    # 构建掩码
    mask = (target != padding_value).float()  # 填充值为 -1
    # 计算损失
    loss = F.mse_loss(output, target, reduction='none')  # 不对损失进行归一化
    # 忽略填充值
    masked_loss = (loss * mask).sum() / mask.sum()
    return masked_loss
