from .basic_module import BasicModule

from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.right = shortcut

    def forward(self, X):
        out = self.conv1(X)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.right(X) if self.right else X
        out += residual
        out = F.relu(out)
        return out


class ResNet34(BasicModule):

    def __init__(self, num_classes=2):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self._make_layer(64, 64, 3, 1, False)
        self.layer2 = self._make_layer(64, 128, 4, 2, True)
        self.layer3 = self._make_layer(128, 256, 6, 2, True)
        self.layer4 = self._make_layer(256, 512, 3, 2, True)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, block_num, stride, is_shortcut):
        shortcut = None
        if is_shortcut:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        rb = ResidualBlock(in_channels, out_channels, stride, shortcut)
        layers.append(rb)
        for i in range(1, block_num):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, X):
        out = self.pre(X)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


