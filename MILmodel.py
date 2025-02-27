import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from nystrom_attention import NystromAttention
from torch.nn import MultiheadAttention
import torch
import torch.nn as nn
import math

class RoPE(nn.Module):
    def __init__(self, dim, max_len=50000):
        super(RoPE, self).__init__()
        self.dim = dim
        self.max_len = max_len
        
        # 预先计算出旋转因子
        inv_freq = 1.0 / (10000**(torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_len).float().unsqueeze(1)
        sinusoid = torch.matmul(position, inv_freq.unsqueeze(0))  # [max_len, dim//2]
        
        self.register_buffer("sinusoid", sinusoid)

    def forward(self, x):
        # x: [B, N, d_model]
        B, N, _ = x.size()
        
        # 获取旋转的位置编码
        sinusoids = self.sinusoid[:N, :]  # [N, dim//2]
        sinusoids = torch.cat((sinusoids, sinusoids), dim=-1)  # [N, dim]
        
        # 将位置编码应用到输入 x 上
        return x * torch.cos(sinusoids) + x.roll(1, dims=1) * torch.sin(sinusoids)
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_rate=0.1, num_landmarks=256):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = NystromAttention(
            dim=hidden_dim,
            dim_head=hidden_dim // num_heads,
            heads=num_heads,
            num_landmarks=num_landmarks,
            dropout=dropout_rate
        )
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.activation = F.relu

    def forward(self, src, src_mask=None):
        # Self-Attention
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Feed-Forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_rate=0.1, num_landmarks=256):
        super(CustomTransformerDecoderLayer, self).__init__()
        self.self_attn = NystromAttention(
            dim=hidden_dim,
            dim_head=hidden_dim // num_heads,
            heads=num_heads,
            num_landmarks=num_landmarks,
            dropout=dropout_rate
        )
        self.multihead_attn = MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.activation = F.relu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-Attention
        tgt2 = self.self_attn(tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # Cross-Attention
        tgt2, _ = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # # Feed-Forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, N, d_model]
        B, N, _ = x.size()
        x = x + self.pe[:, :N]
        return x



class MILModel(nn.Module):
    def __init__(
        self,
        num_classes=5,
        num_instances=5,
        feature_dim=1024,
        hidden_dim=512,
        num_layers=1,
        max_len=35000,
        dropout_rate=0.1,
        num_landmarks=256
    ):
        super(MILModel, self).__init__()
        self.num_classes = num_classes
        self.num_instances = num_instances

        # 特征降维
        self.fc_reduce = nn.Linear(feature_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 位置编码
        self.pos_encoder = RoPE(dim=hidden_dim, max_len=max_len)

        # 编码器
        encoder_layers = nn.ModuleList([CustomTransformerEncoderLayer(hidden_dim, num_heads=8, dropout_rate=dropout_rate, num_landmarks=num_landmarks) for _ in range(num_layers)])
        self.encoder = nn.Sequential(*encoder_layers)

        # 解码器
        decoder_layers = nn.ModuleList([CustomTransformerDecoderLayer(hidden_dim, num_heads=8, dropout_rate=dropout_rate, num_landmarks=num_landmarks) for _ in range(2)])
        self.decoder = nn.Sequential(*decoder_layers)

        # 解码器的输入（查询）嵌入
        self.query_embed = nn.Embedding(num_instances, hidden_dim)

        # ABMIL 门控注意力机制
        self.attn_weight = nn.Linear(hidden_dim, 1)

        # 1D 卷积层，用于消融实验（保持注释状态）
        self.conv1d = nn.Conv1d(in_channels=hidden_dim, out_channels=5, kernel_size=5, padding=0, stride=1)
        # 1D 卷积层，用于消融实验（保持注释状态）
        # self.conv1d = nn.Conv1d(
        #     in_channels=hidden_dim, 
        #     out_channels=1, 
        #     kernel_size=5, 
        #     padding=2, 
        #     stride=1
        # )
        # 分类层
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, mask=None):
        # 特征降维
        x = self.fc_reduce(x)  # [B, N, hidden_dim]
        x = self.layer_norm(x)

        # 添加 RoPE 位置编码
        x = self.pos_encoder(x)  # [B, N, hidden_dim]

        # 编码器
        memory = x
        for layer in self.encoder:
            memory = layer(memory, src_mask=mask)  # [B, N, hidden_dim]

        # 解码器的输入（查询），形状为 [B, num_instances, hidden_dim]
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, 5, hidden_dim]

        # 解码器
        hs = query_embed
        for layer in self.decoder:
            hs = layer(hs, memory, tgt_mask=None, memory_mask=mask)  # [B, 5, hidden_dim]

        # 特征融合使用 ABMIL 的门控注意力
        weights = self.attn_weight(hs)  # [B, 5, 1]
        weights = F.softmax(weights, dim=1)  # [B, 5, 1]
        hs = hs * weights  # [B, 5, hidden_dim]
        hs = hs.sum(dim=1)  # [B, hidden_dim]
        # 替代的特征融合方式：1D 卷积层（用于消融实验）
        # hs_conv = hs.permute(0, 2, 1)  # [B, hidden_dim, 5]
        # hs_conv = self.conv1d(hs_conv)  # [B, 1, 5]
        # hs_conv = hs_conv.squeeze(1)  # [B, 5]
        # logits = hs_conv  # [B, 5]
        # hs_conv = hs.permute(0, 2, 1)  # [B, 512, 5]
        # hs_conv = self.conv1d(hs_conv)  # [B, 5, 1]
        # hs_conv = hs_conv.squeeze(-1)   # [B, 5]
        # logits = hs_conv
        # 分类层
        logits = self.classifier(hs)  # [B, 5]
        return logits
