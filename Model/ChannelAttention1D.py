import torch
import torch.nn as nn

class ChannelAttention1D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

# 示例用法：
if __name__ == '__main__':
    # 假设输入数据的维度为 (batch_size, in_channels, sequence_length)
    input_data = torch.randn(1, 16, 256)  # 示例输入数据
    ca = ChannelAttention1D(16)  # 创建通道注意力模块
    output = ca(input_data)  # 应用通道注意力
    print(output.shape)  # 输出的形状应与输入相同
