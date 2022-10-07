'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10, scale=1):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, int(scale*192), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(scale*192)),
            nn.ReLU(True),
        )

        self.a3 = Inception(int(scale*192),  int(scale*64),  int(scale*96), int(scale*128), int(scale*16), int(scale*32), int(scale*32))
        self.b3 = Inception(int(scale*256), int(scale*128), int(scale*128), int(scale*192), int(scale*32), int(scale*96), int(scale*64))

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(int(scale*480), int(scale*192),  int(scale*96), int(scale*208), int(scale*16),  int(scale*48),  int(scale*64))
        self.b4 = Inception(int(scale*512), int(scale*160), int(scale*112), int(scale*224), int(scale*24),  int(scale*64),  int(scale*64))
        self.c4 = Inception(int(scale*512), int(scale*128), int(scale*128), int(scale*256), int(scale*24),  int(scale*64),  int(scale*64))
        self.d4 = Inception(int(scale*512), int(scale*112), int(scale*144), int(scale*288), int(scale*32),  int(scale*64),  int(scale*64))
        self.e4 = Inception(int(scale*528), int(scale*256), int(scale*160), int(scale*320), int(scale*32), int(scale*128), int(scale*128))

        self.a5 = Inception(int(scale*832), int(scale*256), int(scale*160), int(scale*320), int(scale*32), int(scale*128), int(scale*128))
        self.b5 = Inception(int(scale*832), int(scale*384), int(scale*192), int(scale*384), int(scale*48), int(scale*128), int(scale*128))

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(int(scale*1024), num_classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = GoogLeNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()
