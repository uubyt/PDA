# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import TransformerModel
from data_processing import process_data
from train import train_model, save_model
from test import test_model, load_model

# 主函数：加载数据、初始化模型、训练和评估
# 主函数：加载数据、预处理、划分数据集
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    file_path = 'G:\Test\PDA\dataset\pda_data.csv'
    (X_train, X_test, y_train, y_test, mask_train, mask_test), max_seq_len = process_data(file_path)


    src_input_dim = 6
    tgt_input_dim = max_seq_len
    embed_dim = 32
    num_heads = 4
    num_layers = 2
    output_dim = max_seq_len
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.001

    # 创建训练和测试数据集
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train, mask_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test, mask_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(src_input_dim, tgt_input_dim, embed_dim, num_heads, num_layers, max_seq_len, output_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}')
     # 保存模型权重
    save_model(model, "model_weights.pth")
    # 加载模型权重
    load_model(model, "model_weights.pth")
     # 测试模型
    test_loss, output = test_model(model, test_loader, device)
    print(f'Epoch {epoch + 1}/{num_epochs}, Test Loss: {test_loss:.4f}')
    print(f'output:{output}')
if __name__ == "__main__":
    main()
