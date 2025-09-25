import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GroupNorm(nn.Module):
    """组归一化层"""

    def __init__(self, num_channels, num_groups=32):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)
        # 可训练参数gamma用于测量方差
        self.gamma = nn.Parameter(torch.ones(num_channels))

    def forward(self, x):
        x = self.gn(x)
        return x


class SRU(nn.Module):
    """空间重构单元 (Spatial Reconstruction Unit)"""

    def __init__(self, channels):
        super(SRU, self).__init__()
        self.gn = GroupNorm(channels)
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        # 应用组归一化
        x_norm = self.gn(x)

        # 计算每个通道的权重
        B, C, H, W = x.shape
        gamma_expand = self.gamma.view(1, C, 1, 1).expand_as(x_norm)

        # 计算归一化权重
        gamma_sum = torch.sum(self.gamma)
        weights = gamma_expand / gamma_sum  # W_γ

        # 缩放到[0,1]范围
        weights_scaled = torch.sigmoid(weights)

        # 阈值处理
        threshold = 0.5
        W1 = (weights_scaled > threshold).float()
        W2 = (weights_scaled <= threshold).float()

        # 加权特征
        X_W1 = x * W1
        X_W2 = x * W2

        # 交叉重构
        return torch.cat([X_W1, X_W2], dim=1)


class CRU(nn.Module):
    """通道重构单元 (Channel Reconstruction Unit)"""

    def __init__(self, channels, split_ratio=0.5, reduction=4, kernel_size=3):
        super(CRU, self).__init__()
        self.split_ratio = split_ratio
        self.channels = channels

        # 计算分割后的通道数
        self.upper_channels = int(channels * split_ratio)
        self.lower_channels = channels - self.upper_channels

        # 1x1压缩卷积 - 修复：去掉多余的 * 2
        self.compress = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)

        # 上半部分 - 丰富特征处理
        self.gwc = nn.Conv2d(self.upper_channels, self.upper_channels,
                             kernel_size, 1, kernel_size // 2,
                             groups=self.upper_channels, bias=False)
        self.pwc1 = nn.Conv2d(self.upper_channels, self.upper_channels, 1, 1, 0, bias=False)

        # 下半部分 - 浅层特征处理
        self.pwc2 = nn.Conv2d(self.lower_channels, self.lower_channels, 1, 1, 0, bias=False)

        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 通道注意力
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        # 输入压缩
        x = self.compress(x)

        # 分割特征
        x_upper = x[:, :self.upper_channels, :, :]
        x_lower = x[:, self.upper_channels:, :, :]

        # 上半部分处理 - 组卷积 + 点卷积
        y1 = self.gwc(x_upper) + self.pwc1(x_upper)

        # 下半部分处理 - 仅点卷积
        y2 = self.pwc2(x_lower) + x_lower

        # 合并特征
        y = torch.cat([y1, y2], dim=1)

        # 全局特征信息
        b, c, h, w = y.size()
        s = self.avg_pool(y).view(b, c)

        # 通道软注意力
        s_compressed = F.relu(self.fc1(s))
        attention = torch.sigmoid(self.fc2(s_compressed)).view(b, c, 1, 1)

        return y * attention


class SCConv(nn.Module):
    """空间和通道重构卷积 (Spatial and Channel Reconstruction Convolution)"""

    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super(SCConv, self).__init__()
        self.sru = SRU(channels)
        self.cru = CRU(channels * 2)  # SRU输出通道数翻倍

    def forward(self, x):
        # 空间重构
        x = self.sru(x)
        # 通道重构
        x = self.cru(x)
        return x


class ECA(nn.Module):
    """高效通道注意力 (Efficient Channel Attention)"""

    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        # 自适应确定1D卷积核大小
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化
        y = self.avg_pool(x)

        # 1D卷积 (跨通道交互)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # 注意力权重
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SCECA(nn.Module):
    """SCECA模块：SCConv + ECA注意力"""

    def __init__(self, channels):
        super(SCECA, self).__init__()
        self.scconv = SCConv(channels)
        # ECA应该接收SCConv输出的通道数（翻倍后）
        self.eca = ECA(channels * 2)
        # 添加1x1卷积将通道数调整回原来的大小
        self.channel_adjust = nn.Conv2d(channels * 2, channels, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.scconv(x)
        x = self.eca(x)
        # 调整通道数回到原始大小
        x = self.channel_adjust(x)
        return x


class DenseLayer(nn.Module):
    """密集层 - 减少特征冗余"""

    def __init__(self, channels, growth_rate=32, num_layers=4, dropout_rate=0.5):
        super(DenseLayer, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # 密集连接的卷积层
        self.dense_convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = channels + i * growth_rate
            self.dense_convs.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels, growth_rate, 3, 1, 1, bias=False)
                )
            )

        self.dropout = nn.Dropout2d(dropout_rate)

        # 输出调整层
        final_channels = channels + num_layers * growth_rate
        self.transition = nn.Conv2d(final_channels, channels, 1, 1, 0, bias=False)

    def forward(self, x):
        features = [x]

        for i in range(self.num_layers):
            # 连接之前的所有特征
            concat_features = torch.cat(features, dim=1)
            # 卷积处理
            new_feature = self.dense_convs[i](concat_features)
            # Dropout
            new_feature = self.dropout(new_feature)
            features.append(new_feature)

        # 最终特征连接和转换
        final_features = torch.cat(features, dim=1)
        output = self.transition(final_features)

        return output


class DoubleConv(nn.Module):
    """双卷积层"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SCECANet(nn.Module):
    """SCECA-Net主模型"""

    def __init__(self, in_channels=10, out_channels=5, features=[64, 128, 256, 512]):
        super(SCECANet, self).__init__()
        self.features = features

        # 编码器部分
        self.encoder_convs = nn.ModuleList()
        self.sceca_modules = nn.ModuleList()
        self.pools = nn.ModuleList()

        # 第一层
        self.encoder_convs.append(DoubleConv(in_channels, features[0]))
        self.sceca_modules.append(SCECA(features[0]))

        # 后续编码器层
        for i in range(1, len(features)):
            self.encoder_convs.append(DoubleConv(features[i - 1], features[i]))
            self.sceca_modules.append(SCECA(features[i]))
            self.pools.append(nn.MaxPool2d(2))

        # 底层Dense Layer
        self.dense_layer = DenseLayer(features[-1])

        # 解码器部分
        self.decoder_convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in range(len(features) - 1, 0, -1):
            self.upsamples.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            self.decoder_convs.append(DoubleConv(features[i] + features[i - 1], features[i - 1]))

        # 输出层
        self.output_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        # 编码器前向传播
        skip_connections = []

        for i in range(len(self.features)):
            # 双卷积
            x = self.encoder_convs[i](x)
            # SCECA特征重组
            x_sceca = self.sceca_modules[i](x)
            skip_connections.append(x_sceca)

            # 除了最后一层，都进行池化，池化SCECA的输出
            if i < len(self.features) - 1:
                x = self.pools[i](x_sceca)
            else:
                x = x_sceca  # 最后一层使用SCECA的输出

        # 底层Dense Layer处理
        x = self.dense_layer(x)

        # 解码器前向传播
        skip_connections.reverse()  # 反向使用跳跃连接

        for i in range(len(self.decoder_convs)):
            # 上采样
            x = self.upsamples[i](x)

            # 与跳跃连接的特征融合
            skip_feature = skip_connections[i + 1]  # +1因为最底层不参与跳跃连接

            # 确保尺寸匹配
            if x.shape[2:] != skip_feature.shape[2:]:
                x = F.interpolate(x, size=skip_feature.shape[2:], mode='bilinear', align_corners=True)

            # 连接特征
            x = torch.cat([skip_feature, x], dim=1)

            # 双卷积处理
            x = self.decoder_convs[i](x)

        # 输出预测
        output = self.output_conv(x)

        return output


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型实例
    model = SCECANet(in_channels=5, out_channels=20).to(device)

    # 打印模型信息
    print(f"模型参数数量: {count_parameters(model):,}")
    print("\n模型结构:")
    print(model)

    # 测试前向传播
    batch_size = 1
    input_tensor = torch.randn(batch_size, 5, 256, 256).to(device)

    print(f"\n输入形状: {input_tensor.shape}")

    with torch.no_grad():
        output = model(input_tensor)
        print(f"输出形状: {output.shape}")

    print("\n模型测试成功！")