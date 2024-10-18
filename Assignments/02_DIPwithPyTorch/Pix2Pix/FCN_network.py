import torch.nn as nn
from torch import cat

class BasicBlock(nn.Module):
    def __init__(self, channel) -> None:
        super(BasicBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = self.relu(out)
        return out

class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        out = self.layer(x)
        return out
    
class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super(UpSample, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class FullyConvNetwork(nn.Module):
    def __init__(self):
        super(FullyConvNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            DownSample(3, 64),
            BasicBlock(64),
            BasicBlock(64),
        )
        self.layer2 = nn.Sequential(
            DownSample(64, 128),
            BasicBlock(128),
            BasicBlock(128),
        )
        self.layer3 = nn.Sequential(
            DownSample(128, 256),
            BasicBlock(256),
            BasicBlock(256),
        )
        self.layer4 = nn.Sequential(
            DownSample(256, 512),
            BasicBlock(512),
            BasicBlock(512),
        )
        self.layer5 = nn.Sequential(
            DownSample(512, 1024),
            BasicBlock(1024),
            BasicBlock(1024),
        )
        self.upsample1 = nn.Sequential(
            UpSample(1024, 512),
            BasicBlock(512),
            BasicBlock(512),
        )
        self.upsample2 = nn.Sequential(
            UpSample(512, 256),
            BasicBlock(256),
            BasicBlock(256),
        )
        self.upsample3 = nn.Sequential(
            UpSample(256, 128),
            BasicBlock(128),
            BasicBlock(128),
        )
        self.upsample4 = nn.Sequential(
            UpSample(128, 64),
            BasicBlock(64),
            BasicBlock(64),
        )
        self.upsample5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()
 
    def forward(self, x):
        d1 = self.layer1(x)
        d2 = self.layer2(d1)
        d3 = self.layer3(d2)
        d4 = self.layer4(d3)
        d5 = self.layer5(d4)
        u1 = self.upsample1(d5) + d4
        u2 = self.upsample2(u1) + d3 
        u3 = self.upsample3(u2) + d2 
        u4 = self.upsample4(u3) + d1
        u5 = self.upsample5(u4)
        out = self.tanh(u5)
        return out