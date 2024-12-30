import torch
import torch.nn as nn
from utils import compute_loss
from train import create_padding_mask
from train import create_causal_mask

# 测试函数与训练类似，可以参考train_model函数进行修改
def test_model(model, test_loader, device):
    model.eval()
    total_loss = 0
    all_predictions = []  # 用于存储所有预测值
    all_targets = []  # 用于存储所有目标值

    with torch.no_grad():
        for batch_features, batch_outputs, batch_masks in test_loader:
            batch_features, batch_outputs, batch_masks = batch_features.to(device), batch_outputs.to(device), batch_masks.to(device)
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
            total_loss += loss.item()

        return total_loss / len(test_loader), output


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()  # 设置为评估模式
    return model