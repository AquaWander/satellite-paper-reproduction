# -*- coding: utf-8 -*-
"""
神经网络模型模块
实现三种模型:
  1. TD-CNN-LSTM (本文提出方法) - Split Learning架构
  2. CNN-LSTM [7] (基线1)
  3. LSTM-only [29] (基线2)

使用PyTorch实现
论文: Split Learning-Based Channel Prediction for 6G-Enabled LEO Satellite Systems

注意: 为了在CPU上也能快速训练，使用池化层降维
"""

import torch
import torch.nn as nn
import numpy as np
from config import (
    N_TIME_STEPS, N_DMRS, N_SUBCARRIERS, N_CH, OUTPUT_DIM,
    CONV1_FILTERS, CONV2_FILTERS, CONV_KERNEL_SIZE,
    LSTM_UNITS, DROPOUT_RATE, DENSE1_UNITS,
)


class Flatten(nn.Module):
    """展平层"""
    def forward(self, x):
        return x.reshape(x.size(0), -1)


# ============================================================
# 模型1: TD-CNN-LSTM (本文提出方法) - Split Learning架构
# ============================================================

class SM_LEO(nn.Module):
    """
    卫星端子模型 (Split Learning的客户端)
    结构: TD-Conv2D(16, 3x3) -> LayerNorm -> TD-Conv2D(32, 3x3) -> LayerNorm
          -> AvgPool -> Flatten (cut layer)

    "TD" (Time-Distributed): 对每个时间步独立应用2D卷积
    即输入 (batch, N_U, N_p, N_SC, N_CH) 中 N_U 维度上共享参数

    使用池化层降维以加速LSTM计算
    """
    def __init__(self):
        super(SM_LEO, self).__init__()

        # 第一个时域分布卷积层: 16个3x3滤波器
        self.conv1 = nn.Conv2d(
            in_channels=N_CH,
            out_channels=CONV1_FILTERS,
            kernel_size=(CONV_KERNEL_SIZE, CONV_KERNEL_SIZE),
            padding=1
        )
        self.ln1 = nn.LayerNorm([N_DMRS, N_SUBCARRIERS, CONV1_FILTERS])

        # 第二个时域分布卷积层: 32个3x3滤波器
        self.conv2 = nn.Conv2d(
            in_channels=CONV1_FILTERS,
            out_channels=CONV2_FILTERS,
            kernel_size=(CONV_KERNEL_SIZE, CONV_KERNEL_SIZE),
            padding=1
        )
        self.ln2 = nn.LayerNorm([N_DMRS, N_SUBCARRIERS, CONV2_FILTERS])

        # 池化层: 在子载波维度降维 (300 -> 15), 保持N_p维度
        self.pool = nn.AvgPool2d(kernel_size=(1, 20))  # (N_p, N_SC) -> (N_p, N_SC/20)

        self.flatten = Flatten()

        # Flatten后的维度: CONV2_FILTERS * N_DMRS * (N_SUBCARRIERS/20)
        self.pooled_sc = N_SUBCARRIERS // 20  # = 15
        self.cut_dim = CONV2_FILTERS * N_DMRS * self.pooled_sc  # 32*2*15 = 960

    def forward(self, x):
        """
        输入: (batch, N_U, N_p, N_SC, N_CH)
        时域分布处理: 对每个时间步应用相同的Conv2D
        """
        batch_size, n_u = x.size(0), x.size(1)

        # 重塑为 (batch*N_U, N_CH, N_p, N_SC) 以实现时域分布
        x = x.reshape(-1, N_CH, N_DMRS, N_SUBCARRIERS)

        # Conv Block 1
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C) for LayerNorm
        x = self.ln1(x)
        x = x.permute(0, 3, 1, 2)  # 恢复 (N, C, H, W)
        x = torch.relu(x)

        # Conv Block 2
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ln2(x)
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(x)

        # 池化降维
        x = self.pool(x)  # (batch*N_U, 32, N_p, pooled_sc)

        # Flatten
        x = self.flatten(x)  # (batch*N_U, cut_dim)

        # 恢复时间维度
        x = x.reshape(batch_size, n_u, -1)  # (batch, N_U, cut_dim)

        return x


class SM_UE(nn.Module):
    """
    用户端子模型 (Split Learning的服务端)
    结构: LSTM(512)x2 -> Dropout(0.25) -> Dense(1024, LeakyReLU) -> Dense(N_p*N_SC*N_CH)

    接收SM_LEO在cut layer的输出，完成剩余计算
    """
    def __init__(self, input_dim):
        super(SM_UE, self).__init__()

        # 双层LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=LSTM_UNITS,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        self.dropout = nn.Dropout(DROPOUT_RATE)

        # Dense层
        self.dense1 = nn.Linear(LSTM_UNITS, DENSE1_UNITS)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dense2 = nn.Linear(DENSE1_UNITS, OUTPUT_DIM)

    def forward(self, x):
        """
        输入: SM_LEO的输出 (batch, N_U, cut_dim)
        """
        # LSTM提取时序特征
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch, LSTM_UNITS)

        # Dropout
        out = self.dropout(last_output)

        # Dense层
        out = self.dense1(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        out = self.dense2(out)

        return out


class TDCNNLSTM(nn.Module):
    """
    完整的TD-CNN-LSTM模型 (Split Learning)
    SM_LEO (卫星端) + SM_UE (用户端) 通过cut layer连接

    这是论文提出的完整架构
    """
    def __init__(self):
        super(TDCNNLSTM, self).__init__()

        # 卫星端子模型
        self.sm_leo = SM_LEO()
        cut_dim = self.sm_leo.cut_dim  # 960

        # 用户端子模型
        self.sm_ue = SM_UE(input_dim=cut_dim)

    def forward(self, x):
        """
        输入: (batch, N_U, N_p, N_SC, N_CH)
        输出: (batch, N_p*N_SC*N_CH)
        """
        # SM_LEO处理 (cut layer之前)
        features = self.sm_leo(x)

        # SM_UE处理 (cut layer之后)
        output = self.sm_ue(features)

        return output

    def get_cut_layer_output(self, x):
        """获取cut layer的输出 (用于Split Learning通信模拟)"""
        return self.sm_leo(x)


# ============================================================
# 模型2: CNN-LSTM [7] (基线1 - 无Time-Distributed)
# ============================================================

class CNNLSTMBaseline(nn.Module):
    """
    CNN-LSTM基线模型 (参考文献[7])
    与TD-CNN-LSTM的区别: 不使用Time-Distributed卷积
    而是先将所有时间步展平后统一处理

    结构: Conv2D -> Conv2D -> Pool -> Flatten -> LSTM -> Dense
    """
    def __init__(self):
        super(CNNLSTMBaseline, self).__init__()

        # 将(N_U, N_p, N_SC, N_CH)展平后直接Conv
        in_channels = N_CH * N_TIME_STEPS  # 2*5=10

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=CONV1_FILTERS,
            kernel_size=(CONV_KERNEL_SIZE, CONV_KERNEL_SIZE),
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(CONV1_FILTERS)

        self.conv2 = nn.Conv2d(
            in_channels=CONV1_FILTERS,
            out_channels=CONV2_FILTERS,
            kernel_size=(CONV_KERNEL_SIZE, CONV_KERNEL_SIZE),
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(CONV2_FILTERS)

        # 池化降维
        self.pool = nn.AvgPool2d(kernel_size=(1, 20))
        pooled_sc = N_SUBCARRIERS // 20  # 15
        conv_flat_dim = CONV2_FILTERS * N_DMRS * pooled_sc  # 32*2*15=960

        # 单层LSTM (比proposed少一层)
        self.lstm = nn.LSTM(
            input_size=conv_flat_dim,
            hidden_size=LSTM_UNITS // 2,  # 256
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.dense1 = nn.Linear(LSTM_UNITS // 2, DENSE1_UNITS // 2)  # 512
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dense2 = nn.Linear(DENSE1_UNITS // 2, OUTPUT_DIM)

    def forward(self, x):
        """
        输入: (batch, N_U, N_p, N_SC, N_CH)
        """
        batch_size = x.size(0)

        # 展平时间维度到通道维度
        x = x.reshape(batch_size, N_TIME_STEPS * N_CH, N_DMRS, N_SUBCARRIERS)

        # CNN特征提取
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))

        # 池化
        x = self.pool(x)

        # Flatten
        x = x.reshape(batch_size, 1, -1)  # (batch, 1, conv_flat_dim)

        # LSTM
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]

        # Dense
        out = self.dropout(out)
        out = self.leaky_relu(self.dense1(out))
        out = self.dropout(out)
        out = self.dense2(out)

        return out


# ============================================================
# 模型3: LSTM-only [29] (基线2 - 无CNN特征提取)
# ============================================================

class LSTMOnlyBaseline(nn.Module):
    """
    LSTM-only基线模型 (参考文献[29])
    仅使用LSTM处理原始信道数据，无CNN特征提取

    结构: Flatten -> LSTM(256)x2 -> Dense -> Dense
    """
    def __init__(self):
        super(LSTMOnlyBaseline, self).__init__()

        # 每个时间步的输入维度: N_p * N_SC * N_CH = 1200
        self.input_dim = N_DMRS * N_SUBCARRIERS * N_CH

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=LSTM_UNITS // 2,  # 256
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.dense1 = nn.Linear(LSTM_UNITS // 2, DENSE1_UNITS // 2)  # 512
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dense2 = nn.Linear(DENSE1_UNITS // 2, OUTPUT_DIM)

    def forward(self, x):
        """
        输入: (batch, N_U, N_p, N_SC, N_CH)
        """
        batch_size = x.size(0)

        # 展平每个时间步的空间维度
        x = x.reshape(batch_size, N_TIME_STEPS, -1)  # (batch, N_U, 1200)

        # LSTM
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # 最后时间步

        # Dense
        out = self.dropout(out)
        out = self.leaky_relu(self.dense1(out))
        out = self.dropout(out)
        out = self.dense2(out)

        return out


# ============================================================
# NMSE损失函数
# ============================================================

class NMSELoss(nn.Module):
    """
    NMSE损失函数 (论文Eq.13)
    L = (1/N_B) * sum ||H_tilde_OUT - H_OUT||^2 / ||H_tilde_OUT||^2
    """
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, predicted, target):
        """
        predicted: (batch, D)
        target: (batch, D)
        """
        numerator = torch.sum((predicted - target) ** 2, dim=1)
        denominator = torch.sum(target ** 2, dim=1) + 1e-10

        nmse = torch.mean(numerator / denominator)
        return nmse


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("=" * 60)
    print("模型结构测试")
    print("=" * 60)

    # 测试输入
    batch_size = 8
    x = torch.randn(batch_size, N_TIME_STEPS, N_DMRS, N_SUBCARRIERS, N_CH)
    y = torch.randn(batch_size, OUTPUT_DIM)

    # 模型1: TD-CNN-LSTM (Proposed)
    model1 = TDCNNLSTM()
    out1 = model1(x)
    print(f"\n[Proposed] TD-CNN-LSTM:")
    print(f"  输入: {x.shape}")
    print(f"  输出: {out1.shape}")
    print(f"  参数量: {count_parameters(model1):,}")
    print(f"  SM_LEO参数量: {count_parameters(model1.sm_leo):,}")
    print(f"  SM_UE参数量: {count_parameters(model1.sm_ue):,}")
    print(f"  Cut dim: {model1.sm_leo.cut_dim}")

    # 模型2: CNN-LSTM [7]
    model2 = CNNLSTMBaseline()
    out2 = model2(x)
    print(f"\n[基线1] CNN-LSTM [7]:")
    print(f"  输入: {x.shape}")
    print(f"  输出: {out2.shape}")
    print(f"  参数量: {count_parameters(model2):,}")

    # 模型3: LSTM-only [29]
    model3 = LSTMOnlyBaseline()
    out3 = model3(x)
    print(f"\n[基线2] LSTM-only [29]:")
    print(f"  输入: {x.shape}")
    print(f"  输出: {out3.shape}")
    print(f"  参数量: {count_parameters(model3):,}")

    # 测试NMSE损失
    criterion = NMSELoss()
    loss1 = criterion(out1, y)
    print(f"\nNMSE损失测试: {loss1.item():.4f}")
    print(f"NMSE (dB): {10*np.log10(loss1.item() + 1e-10):.2f} dB")
