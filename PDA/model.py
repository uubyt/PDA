import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, src_input_dim, tgt_input_dim, embed_dim, num_heads, num_layers, max_seq_len, output_dim):
        super(TransformerModel, self).__init__()
        self.src_embedding = nn.Linear(src_input_dim, embed_dim)
        self.tgt_embedding = nn.Linear(tgt_input_dim, embed_dim)
        # 位置编码
        
        #self.positional_encoding = self._generate_positional_encoding(max_seq_len, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        # Transformer 编码器和解码器
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def _generate_positional_encoding(self, max_len, embed_dim):
        """生成位置编码"""
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 增加 batch 维度
        return pe

    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # 嵌入 + 位置编码
        print("src_key_padding_mask.shape:", src_key_padding_mask.shape)
        print("tgt_key_padding_mask.shape:", tgt_key_padding_mask.shape)
        src = self.src_embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt = self.tgt_embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        # Transformer 编码器和解码器
        memory = self.encoder(src.permute(1, 0, 2), src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(
            tgt.permute(1, 0, 2), 
            memory, 
            memory_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        output = self.fc_out(output.permute(1, 0, 2))  # 转回 Batch-First
        return output

