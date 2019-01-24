import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class Bottleneck(nn.Module):
    """Bottleneck Unit"""
    def __init__(self, in_c, mid_c, out_c, stride=1, padding=0, downsample=None):
        super(Bottleneck, self).__init__()
        self.reduce = nn.Sequential(nn.Conv3d(in_c, mid_c, kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm3d(mid_c),
                                       nn.ReLU(inplace=True),)
        self.conv3 = nn.Sequential(nn.Conv3d(mid_c, mid_c, kernel_size=3, stride=stride, padding=padding),
                                       nn.BatchNorm3d(mid_c),
                                       nn.ReLU(inplace=True),)
        self.increase = nn.Sequential(nn.Conv3d(mid_c, out_c, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm3d(out_c))

        self.downsample = downsample
        if self.downsample:
            self.proj = nn.Sequential(nn.Conv3d(in_c, out_c, 1, stride=stride, padding=0, dilation=1),
                                      nn.BatchNorm3d(out_c))
    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3(h)
        h = self.increase(h)
        if self.downsample:
            h += self.proj(x)
        else:
            h += x
        return F.relu(h)


class ResBlock(nn.Module):
    """Residual Block"""

    def __init__(self, n_layers, in_c, mid_c, out_c, stride, padding):
        super(ResBlock, self).__init__()
        blocks = [Bottleneck(in_c, mid_c, out_c, stride=stride, padding=padding, downsample= True)]
        for i in range(2, n_layers + 1):
            blocks.append(Bottleneck(out_c, mid_c, out_c, stride=1, padding=1, downsample= False))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ConBR(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConBR, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=1, padding=0)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x_res = self.conv2(x)
        x_res = self.relu(x_res)
        x_res = self.conv3(x_res)

        return x+x_res

class GCN(nn.Module):
    def __init__(self, in_c, out_c, k=(7, 7, 3)):  # out_Channel=21 in paper
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv3d(in_c, out_c, kernel_size=(k[0], 1, 1), padding=((k[0] - 1) // 2, 0, 0))
        self.conv_l2 = nn.Conv3d(out_c, out_c, kernel_size=(1, 1, k[2]), padding=(0, 0, (k[2]-1) // 2))
        self.conv_l3 = nn.Conv3d(out_c, out_c, kernel_size=(1, k[1], 1), padding=(0, (k[1] - 1) // 2, 0))
        self.conv_r1 = nn.Conv3d(in_c, out_c, kernel_size=(1, k[1], 1), padding=(0, (k[1] - 1) // 2, 0))
        self.conv_r2 = nn.Conv3d(out_c, out_c, kernel_size=(1, 1, k[2]), padding=(0, 0, (k[2]-1) // 2))
        self.conv_r3 = nn.Conv3d(out_c, out_c, kernel_size=(k[0], 1, 1), padding=((k[0] - 1) // 2, 0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_l = self.conv_l3(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x_r = self.conv_r3(x_r)

        x = x_l + x_r
        return x



class ResBlockMultiGrid(nn.Module):
    """3x Residual Block with multi-grid"""

    def __init__(self, n_layers, in_channels, mid_c, out_c, stride, dilation, mg=[1, 2, 1]):
        super(ResBlockMultiGrid, self).__init__()
        self.blocks = nn.Sequential(
            Bottleneck(in_channels, mid_channels, out_c, stride, dilation * mg[0], True),
            Bottleneck(out_c, mid_channels, out_c, 1, dilation * mg[1], False),
            Bottleneck(out_c, mid_channels, out_c, 1, dilation * mg[2], False)
        )

    def forward(self, x):
        return self.blocks(x)

