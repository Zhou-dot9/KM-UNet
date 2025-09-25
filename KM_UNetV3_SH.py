import torch
import torch.nn as nn
import torch.nn.functional as F
# from kan import KANLinear
# from fast_kan import Fast_KANLinear as KANLinear
from convKAN.KANConv2Dlayers import *
from timm.models.layers import trunc_normal_, DropPath
import torch.quantization
from torch.cuda.amp import autocast
# from torch.amp import autocast, GradScaler
# from torch.nn.functional import ssim

from vim_block_init.efficient_vim_init import EfficientViMBlock
# from ircsa.IRCSA import InterRowColSelfAttention
# from GCA_conv_blk.GCABlock import GCABlock
from WPL.iwp import IntelligentWaveletPoolingModule
from DAGEM_md import DAGEM
from DySample_md import DySample

# ******************** 量化优化组件 ********************
class StableHybridKANConv(nn.Module):
    """稳定性优化的混合KAN卷积模块"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # 基函数扩展为5种
        self.branches = nn.ModuleDict({
            # 'cheby': ChebyKANConv2d(in_channels, out_channels, kernel_size, padding=padding),
            # 'relu': ReLUKANConv2d(in_channels, out_channels, kernel_size, padding=padding),
            # 'wave': WavKANConv2d(in_channels, out_channels, kernel_size, padding=padding),
            # 'rbf': RBFKANConv2d(in_channels, out_channels, kernel_size, padding=padding),
            # 'jacobi': JacobiKANConv2d(in_channels, out_channels, kernel_size, padding=padding)
            'plain': KANConv2d(in_channels, out_channels, kernel_size, padding=padding),
        })#cheby:loss:nan


        self.kanconv2d = nn.Sequential(
            KANConv2d(in_channels, out_channels, kernel_size, padding=padding),
        )
        #modify to followings
        # self.kanconv2d = nn.Sequential(
        #     KANConv2d(in_channels, out_channels, 3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        #     KANConv2d(out_channels, out_channels, 3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
        # 动态注意力权重生成
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, len(self.branches), 1),
            nn.Softmax(dim=1)
        )

        # 稳定性增强组件
        self.pre_norm = nn.GroupNorm(4, in_channels)
        self.post_act = nn.ReLU(inplace=True)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @autocast()
    def forward(self, x):
        x = self.pre_norm(x)
        identity = self.residual(x)

        # # 多分支特征提取
        # branch_outs = []
        # for branch in self.branches.values():
        #     if isinstance(branch, KANLinear):
        #         spatial_feat = x.permute(0, 2, 3, 1)
        #         transformed = branch(spatial_feat).permute(0, 3, 1, 2)
        #     else:
        #         transformed = branch(x)
        #     # print(transformed.shape)
        #     branch_outs.append(transformed)
        #
        # # 注意力融合
        # weights = self.attn(x)  # [B,5,1,1]
        # # print(weights.shape)
        # fused = sum(w * feat for w, feat in zip(weights.unbind(dim=1), branch_outs))

        fused = self.kanconv2d(x)

        return self.post_act(identity + fused)


class EnhancedViMBlock(nn.Module):
    """三维高效ViM模块 - 空间-通道联合建模"""

    def __init__(self, dim, expansion=4, state_dim=64, drop_path=0.1):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim

        # 三个方向处理分支
        self.height_block = DirectionViM(dim, mode='height', state_dim=state_dim)
        self.width_block = DirectionViM(dim, mode='width', state_dim=state_dim)
        self.channel_block = DirectionViM(dim, mode='channel', state_dim=state_dim)

        # 动态融合门控
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 3, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, 3, 1),
            nn.Softmax(dim=1)
        )

        # 前馈增强
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * expansion, 1),
            nn.GELU(),
            nn.Conv2d(dim * expansion, dim, 1)
        )

        # 正则化组件
        self.norm = TripleNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        # print("x.shape before VIM: ",x.shape)
        # 三向特征提取
        h_feat = self.height_block(x)
        # print("h_feat.shape: ",h_feat.shape)
        w_feat = self.width_block(x)
        # print("w_feat.shape: ",w_feat.shape)
        c_feat = self.channel_block(x)
        # print("c_feat.shape: ", c_feat.shape)
        # 动态门控融合
        fusion_weights = self.fusion_gate(torch.cat([h_feat, w_feat, c_feat], dim=1))
        # print("fusion_weights.shape:",fusion_weights.shape)
        fused_feat = fusion_weights[:, 0:1] * h_feat + fusion_weights[:, 1:2] * w_feat + fusion_weights[:, 2:3] * c_feat
        # print("fused_feat.shape:",fused_feat.shape)
        # 残差连接
        x = identity + self.drop_path(fused_feat)

        # 前馈增强
        x = x + self.drop_path(self.ffn(self.norm(x)))

        return x


class DirectionViM(nn.Module):
    """方向专用ViM处理单元"""

    def __init__(self, dim, mode='height', state_dim=64):
        super().__init__()
        self.mode = mode
        self.state_dim = state_dim

        # 状态空间参数
        self.dt_proj = nn.Linear(dim, state_dim)
        # self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.02)
        # self.D = nn.Parameter(torch.randn(dim))
        self.vit_mamba = EfficientViMBlock(dim=dim, mlp_ratio=4, ssd_expand=1, state_dim=64)

        # 方向自适应处理
        if mode == 'height':
            self.proj = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0))
        elif mode == 'width':
            self.proj = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1))
        else:  # channel
            self.proj = nn.Conv2d(dim, dim, 1)

        # 方向注意力
        self.attn = DirectionAttention(dim, mode)

    #    w

    def forward(self, x):
        """方向敏感状态空间扫描"""
        B, C, H, W = x.shape
        # print("x.shape before DIT:",x.shape)
        # 方向投影
        x = self.proj(x)
        # print("x.shape after proj:",x.shape)

        # 按不同方向展开
        # if self.mode == 'height':
        #     x = x.permute(0, 3, 2, 1)  # [B,W,H,C]
        #     seq_dim = 2
        # elif self.mode == 'width':
        #     x = x.permute(0, 2, 1, 3)  # [B,H,C,W]
        #     seq_dim = 3
        # else:  # channel
        #     x = x.permute(0, 2, 3, 1)  # [B,H,W,C]
        #     seq_dim = 1

        # 离散化参数
        # print(x.shape)
        x = self.vit_mamba(x)
        # print("x.shape after VIM:",x.shape)
        # # 恢复形状
        # if self.mode == 'height':
        #     x = x.permute(0, 3, 2, 1)
        # elif self.mode == 'width':
        #     x = x.permute(0, 2, 1, 3)
        # else:
        #     x = x.permute(0, 3, 1, 2)

        return self.attn(x)


class DirectionAttention(nn.Module):
    """方向敏感注意力机制"""

    def __init__(self, dim, mode):
        super().__init__()
        self.mode = mode
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        if mode == 'height':
            self.pool = nn.AdaptiveAvgPool2d((None, 1))
        elif mode == 'width':
            self.pool = nn.AdaptiveAvgPool2d((1, None))
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape#([2, 16, 256, 256])

        # 方向特征聚合
        if self.mode == 'height':
            pooled = self.pool(x)  # [B,C,H,1]
            pooled = pooled.squeeze(-1)  # [B,C,H]
            pooled = pooled.mean(dim=2)  # [B,C]

        elif self.mode == 'width':
            pooled = self.pool(x)  # [B,C,1,W]
            pooled = pooled.squeeze(2)  # [B,C,W]
            pooled = pooled.mean(dim=2)  # [B,C]

        else:  # channel模式
            pooled = self.pool(x)  # [B,C,1,1]
            pooled = pooled.view(B, C)  # [B,C]

        weight = self.fc(pooled.view(B, C))#([2, 16])

        # 局部增强
        q, k, v = self.qkv(x).chunk(3, dim=1)#q:([2, 16, 256, 256]) k:([2, 16, 256, 256]) v:([2, 16, 256, 256])

        attn = torch.sigmoid(q * k) * v#([2, 16, 256, 256])

        return self.conv(attn) * weight.view(B, C, 1, 1)


class TripleNorm(nn.Module):
    """三维联合归一化"""

    def __init__(self, dim):
        super().__init__()
        self.norm_h = nn.GroupNorm(num_groups=1, num_channels=dim)
        # 宽度方向归一化
        self.norm_w = nn.GroupNorm(num_groups=1, num_channels=dim)
        # 通道方向归一化
        self.norm_c = nn.LayerNorm(dim)  # 改用LayerNorm处理通道维度

    def forward(self, x):
        # 高度方向
        h_norm = self.norm_h(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        # 宽度方向
        w_norm = self.norm_w(x)
        # 通道方向
        c_norm = self.norm_c(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return (h_norm + w_norm + c_norm) / 3


class MultiScaleFusion(nn.Module):
    """多尺度特征融合模块"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, channels[-1], s, padding=s//2, stride=1),
                nn.GroupNorm(1, channels[-1]),
                nn.SiLU()
            ) for c, s in zip(channels, [3, 5, 7])
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(channels[-1] * 3, channels[-1], 1),
            nn.Conv2d(channels[-1], channels[-1], 3, padding=1),
            ChannelAttention(channels[-1], reduction)
        )

    @autocast()
    def forward(self, features):
        resized = [block(feat) for block, feat in zip(self.blocks, features)]
        # print(resized[0].shape, resized[1].shape, resized[2].shape)
        fused = self.fusion(torch.cat(resized, dim=1))
        return fused


class ChannelAttention(nn.Module):
    """通道注意力机制"""

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    @autocast()
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# 新增LCA模块
class LocalContrastAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 输入维度: in_channels // reduction_ratio → 输出维度: in_channels
        self.reduction_ratio = reduction_ratio
        self.fc = nn.Sequential(
            nn.Linear(in_channels // reduction_ratio, 64),  # 中间维度
            nn.ReLU(),
            nn.Linear(64, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Step 1: 全局平均池化，保留每个通道的平均值
        # avg = self.avg_pool(x).squeeze()
        avg = self.avg_pool(x).squeeze(-1).squeeze(-1)  # Shape: [N, C]

        # Step 2: 将每个样本的 C 个通道分为 reduction_ratio 组，每组求平均
        avg_reduced = avg.view(avg.size(0), -1, self.reduction_ratio).mean(-1)
        # Shape: [N, C // reduction_ratio]

        # Step 3: 全连接层处理降维后的特征
        global_feat = self.fc(avg_reduced)  # Shape: [N, C]
        # global_feat = torch.sigmoid(global_feat)  # Shape: [N, C]
        # print(f"global_feat shape before expand: {global_feat.shape}")
        # Step 4: 广播权重到原始输入形状
        global_feat = global_feat.unsqueeze(2)  # Shape: [N, C,1]
        global_feat = global_feat.unsqueeze(3)  # Shape: [N, C,1,1]
        # print(f"global_feat shape before expand: {global_feat.shape}")
        global_feat = global_feat.expand(-1, -1, x.size(2), x.size(3))  # torch.Size([4, 16, 256, 256])
        # print(global_feat.shape)
        return x * (1 - global_feat) + global_feat

# ******************** 最终优化模型 ********************
class KM_UNetV3(nn.Module):
    def __init__(self, num_classes=3, embed_dims=[16, 32, 64]):
        super().__init__()

        self.conv_f = nn.Conv2d(5, 16, kernel_size=3, padding=1, stride=1)

        # 编码器
        self.lca1 = LocalContrastAttention(embed_dims[0])
        self.lca2 = LocalContrastAttention(embed_dims[1])
        self.lca3 = LocalContrastAttention(embed_dims[2])



        self.enc1 = nn.Sequential(
            StableHybridKANConv(16, embed_dims[0]),
            # GCABlock(c=embed_dims[0], reduction=2, residual_mode='concat'),  # 或 'add'
            EnhancedViMBlock(embed_dims[0], state_dim=16),
            # nn.MaxPool2d(2)#ke yong IWP
            IntelligentWaveletPoolingModule(embed_dims[0])
        )

        self.enc2 = nn.Sequential(
            StableHybridKANConv(embed_dims[0], embed_dims[1]),
            # MultiScaleFusion([embed_dims[0], embed_dims[1], embed_dims[1]]),
            # GCABlock(c=embed_dims[1], reduction=2, residual_mode='concat'),  # 或 'add'
            EnhancedViMBlock(embed_dims[1], state_dim=16),
            # nn.MaxPool2d(2)
            IntelligentWaveletPoolingModule(embed_dims[1])
        )

        self.enc3 = nn.Sequential(
            StableHybridKANConv(embed_dims[1], embed_dims[2]),
            # GCABlock(c=embed_dims[2], reduction=2, residual_mode='concat'),  # 或 'add'
            EnhancedViMBlock(embed_dims[2], state_dim=16),
            # nn.MaxPool2d(2)
            IntelligentWaveletPoolingModule(embed_dims[2])
        )

        ######################## 新增桥连层 ########################
        # self.bridge_attention = MultiScaleFusion(
        #     [embed_dims[0], embed_dims[1], embed_dims[2]]  # 输入通道配置
        # )
        self.bridge_attention = DAGEM(sync_bn=False, input_channels=embed_dims[2])


        # 解码器
        self.dec1 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DySample(embed_dims[2], scale=2, style='lp'),  # 替换原nn.Upsample
            StableHybridKANConv(embed_dims[2], embed_dims[1])
        )

        self.attention1 = nn.Sequential(
            MultiScaleFusion([embed_dims[0], embed_dims[1], embed_dims[1]]),
        )

        self.attention2 = nn.Sequential(
            MultiScaleFusion([embed_dims[0], embed_dims[1], embed_dims[1]]),
        )

        self.dec2 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DySample(embed_dims[2], scale=2, style='lp'),  # 替换原nn.Upsample
            # MultiScaleFusion([embed_dims[1] * 2, embed_dims[1], embed_dims[0]]),
            nn.Conv2d(embed_dims[1] * 2, embed_dims[1], kernel_size=3, padding=1, stride=1),
            EnhancedViMBlock(embed_dims[1], state_dim=16),
            # StableHybridKANConv(embed_dims[1] * 2, embed_dims[0])
        )

        self.dec3 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DySample(embed_dims[2], scale=2, style='lp'),  # 替换原nn.Upsample
            nn.Conv2d(embed_dims[1] * 2, embed_dims[0], 3, padding=1),
            EnhancedViMBlock(embed_dims[0]),
            nn.Conv2d(embed_dims[0], num_classes, 3, padding=1)
        )

        # 输出层稳定性增强
        self.output_norm = nn.GroupNorm(1, num_classes)
        self.activation = nn.Sigmoid()  # 降雨预测适合Sigmoid输出

        #初始化
    #     self._init_weights()
    #
    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='silu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)

    @autocast()
    def forward(self, x):
        x = self.conv_f(x)
        # print(x.shape)
        # 编码路径
        e1 = self.enc1(x)  # 输出通道数: embed_dims[0] (32)
        # print(e1.shape)
        e1=self.lca1(e1)


        e2 = self.enc2(e1)  # 输出通道数: embed_dims[1] (64)
        # print(e2.shape)
        e2=self.lca2(e2)


        e3 = self.enc3(e2)  # 输出通道数: embed_dims[2] (128)
        # print(e3.shape)
        e3=self.lca3(e3)

        ######################## 桥连层处理 ########################
        # tianjia GEM（桥连层）
        bridge_out = self.bridge_attention(e3)

        # 解码路径
        d1 = self.dec1(bridge_out)  # 输出通道数: embed_dims[1] (64)

        # 生成多尺度特征列表：[e1, e2, e2]，通道数为[32,64,64]
        # 调整特征图尺寸至与d1匹配（假设d1的空间尺寸为H/4, W/4）
        e1_resized = F.interpolate(e1, size=d1.shape[2:], mode='bilinear', align_corners=True)
        e2_resized = F.interpolate(e2, size=d1.shape[2:], mode='bilinear', align_corners=True)
        e3_resized = F.interpolate(e2, size=d1.shape[2:], mode='bilinear', align_corners=True)  # 使用e2重复一次
        # print(e1_resized.shape, e2_resized.shape, e3_resized.shape)
        # 传入多尺度特征列表到attention1
        e2_attn = self.attention1([e1_resized, e2_resized, e3_resized])
        # print(e2_attn.shape)
        d1 = torch.cat([d1, e2_attn], dim=1)
        # print(d1.shape)

        d2 = self.dec2(d1)
        # print(d2.shape)

        # 同理处理attention2
        e1_resized_2 = F.interpolate(e1, size=d2.shape[2:], mode='bilinear', align_corners=True)
        e2_resized_2 = F.interpolate(e2, size=d2.shape[2:], mode='bilinear', align_corners=True)
        e3_resized_2 = F.interpolate(e2, size=d2.shape[2:], mode='bilinear', align_corners=True)

        e1_attn = self.attention2([e1_resized_2, e2_resized_2, e3_resized_2])
        d2 = torch.cat([d2, e1_attn], dim=1)
        # print(d2.shape)

        out = self.dec3(d2)
        out = self.activation(self.output_norm(out))
        return out


# 测试代码
if __name__ == "__main__":
    # 配置混合精度训练
    torch.backends.cudnn.benchmark = True

    # 模型实例化
    model = KM_UNetV3(num_classes=1).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=100000,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    scaler = torch.cuda.amp.GradScaler()

    # 虚拟数据
    dummy_input = torch.randn(4, 5, 64, 64).cuda()
    # print(dummy_input.shape)
    target = torch.rand(4, 1, 64, 64).cuda()

    # 前向测试
    with autocast():
        output = model(dummy_input)
        # print(output)
        print(f"输出形状: {output.shape}")  # 应输出 [4,1,256,256]

        # 损失函数（结合MSE和SSIM）
        mse_loss = F.mse_loss(output, target)
        # ssim_loss = 1 - torch.mean(ssim(output, target, data_range=1.0))
        loss = 0.7 * mse_loss 

    # 反向传播
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # 数值稳定性检查
    assert not torch.isnan(loss).any(), "出现NaN值!"
    print("测试通过，无NaN值产生")