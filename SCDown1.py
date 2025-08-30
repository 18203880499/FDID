import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['SCDown']
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化
        avg_out = torch.mean(x, dim=(2, 3), keepdim=True)
        avg_out = self.fc1(avg_out)
        avg_out = F.relu(avg_out)
        avg_out = self.fc2(avg_out)

        # 通道注意力权重
        max_out = torch.max(x, dim=2, keepdim=True)[0]
        max_out = torch.max(max_out, dim=3, keepdim=True)[0]

        max_out = self.fc1(max_out)
        max_out = F.relu(max_out)
        max_out = self.fc2(max_out)

        # 将平均池化和最大池化的结果相加并经过sigmoid激活
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)  # 生成空间注意力权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 对每个通道进行全局平均池化和最大池化，获得空间特征
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接平均池化和最大池化的结果
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # 计算空间注意力权重
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class SCDown(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention()

        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        # 先应用空间注意力
        x = x * self.spatial_attention(x)

        # 然后应用通道注意力
        x = x * self.channel_attention(x)

        # 下采样
        return self.cv2(self.cv1(x))
