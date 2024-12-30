import os
import torch
import torch.optim as optim
from data_processing import process_data
from model import TransformerModel
from utils import compute_loss


# 训练循环
def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch_features, batch_outputs, batch_masks in train_loader:
        batch_features, batch_outputs, batch_masks = batch_features.to(device), batch_outputs.to(device), batch_masks.to(device)
        optimizer.zero_grad()
        print("batch_masks:", batch_masks)
        print("batch_features:", batch_features)
        print("batch_outputs:", batch_outputs)
        # 将特征复制到序列长度
        src = batch_features.unsqueeze(1).repeat(1, batch_outputs.size(1), 1)
        tgt_padding_mask = create_padding_mask(batch_outputs)
        tgt = batch_outputs.float().unsqueeze(1).repeat(1, batch_outputs.size(1), 1)
        print("src.shape:", src)
        print("tgt.shape:", tgt)
        print("tgt_padding_mask", tgt_padding_mask.shape)
        # 创建掩码
        src_padding_mask = batch_masks  # 源序列的填充掩码来自 batch_masks
        #tgt_padding_mask = create_padding_mask(tgt)
        tgt_causal_mask = create_causal_mask(tgt.size(1)).to(device)
        print("tgt_causal_mask", tgt_causal_mask)
        print("tgt_padding_mask", tgt_padding_mask)
        # 前向传播
        output = model(
            src, tgt,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            src_key_padding_mask=src_padding_mask,  # 传入源序列的填充掩码
        )

        # 计算损失
        output = output.reshape(-1, output.size(-1))
        tgt = tgt.reshape(-1, tgt.size(-1))
        # 计算损失，忽略填充值
        loss = compute_loss(output, tgt, padding_value=-1)
        #loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def create_padding_mask(tgt, padding_value=-1):
    """
    生成掩码，填充值为 `True` 的位置表示无效位置
    """
    return (tgt == padding_value)  # 转换为 (sequence_length, batch_size)


def create_causal_mask(size):
    """
    创建因果掩码：确保模型只关注当前位置及之前的位置信息
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask
