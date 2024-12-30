import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# 加载数据
def load_data(file_path):
    print("Trying to read from:", file_path)
    data = pd.read_csv(file_path)
    return data[:3]

# 数据解析
def parse_data(data):
    features = data[["K1", "F1", "Z1", "S1", "M1", "N1"]].values.astype(np.float32)
    outputs = data["PDA1"].apply(eval).tolist()
    outputs = [[float(y) for y in row] for row in outputs]
    return features, outputs

# 补齐 PDA 输出
def pad_sequence(sequence, max_length, padding_value=-1.0):
    return sequence + [padding_value] * (max_length - len(sequence))

# 生成掩码
def create_mask(seq_len, max_length):
    mask = [1] * seq_len + [0] * (max_length - seq_len)
    return mask


# 划分训练集和测试集
def split_data(features, outputs, masks):
    X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(features, outputs, masks, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, mask_train, mask_test

# 转为 PyTorch 张量
def to_tensor(X_train, X_test, y_train, y_test, mask_train, mask_test):
    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
    # 将掩码列表转换为张量
    mask_train = torch.stack([torch.tensor(mask, dtype=torch.bool) for mask in mask_train])
    mask_test = torch.stack([torch.tensor(mask, dtype=torch.bool) for mask in mask_test])
    return X_train, X_test, y_train, y_test, mask_train, mask_test

# 主函数：加载数据，预处理，划分数据集
def process_data(file_path):
    data = load_data(file_path)
    features, outputs = parse_data(data)
    max_pda_length = max(len(pda) for pda in outputs)
    
    masks = [create_mask(len(pda), max_pda_length) for pda in outputs]
    outputs_padded = [pad_sequence(pda, max_pda_length) for pda in outputs]
    #print("outputs_padded:", outputs_padded)
    X_train, X_test, y_train, y_test, mask_train, mask_test = split_data(features, outputs_padded, masks)
    #print(f"X_train:{X_train}, X_test{X_test}, y_train{y_train}, mask_train{mask_train}, mask_test{mask_test}")
    return to_tensor(X_train, X_test, y_train, y_test, mask_train, mask_test), max_pda_length
