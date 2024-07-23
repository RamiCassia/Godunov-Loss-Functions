import sys
import os
base_path = os.getcwd() + '/'
sys.path.append(base_path)
import torch
import torch.nn as nn


class ConvReLU(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), padding_mode = 'replicate', bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)
        return out

class VDSR(nn.Module):
    def __init__(self) -> None:
        super(VDSR, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(4, 64, (3, 3), (1, 1), (1, 1), padding_mode = 'replicate', bias=False),nn.ReLU(True),)

        trunk = []
        for _ in range(18):
            trunk.append(ConvReLU(64))
        self.trunk = nn.Sequential(*trunk)
        self.conv2 = nn.Conv2d(64, 4, (3, 3), (1, 1), (1, 1), padding_mode = 'replicate', bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.trunk(out)
        out = self.conv2(out)
        out = torch.add(out, identity)

        return out

class Super(nn.Module):
    def __init__(self, in_channels, hidden_channels, scale_factor, mode):
        super(Super, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.scale_factor = scale_factor
        self.mode = mode

        self.convi = nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=3, padding=1, padding_mode = 'replicate')
        self.convo = nn.Conv2d(self.hidden_channels, self.in_channels, kernel_size=3, padding=1, padding_mode = 'replicate')
        self.convh = nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1, padding_mode = 'replicate')
        self.scale_factor = scale_factor
        self.act = nn.LeakyReLU()
        self.act_o = nn.ReLU()
        self.mode = mode
        self.vdsr = VDSR()

    def forward(self, x):

        a,b,c,d,e = x.shape
        x = x.reshape(a*b, c, d, e)
        x = self.act(self.convi(x))

        for i in range(int(self.scale_factor/2)):
            x = self.act(self.convh(x))
            x = self.act(self.convh(x))
            x = self.convh(x)
            x = F.interpolate(x, scale_factor=2, mode=self.mode, align_corners=False)

        x = self.act(self.convh(x))
        x = self.act(self.convh(x))

        x = self.convo(x)
        x = self.vdsr(x)

        x = x.reshape(a,b, c, d*self.scale_factor, e*self.scale_factor)

        x[:, :, [0, 3], :, :] = torch.abs((x[:, :, [0, 3], :, :].clone()))

        return x