from .basic_module import BasicModule

from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.is_shortcut = stride > 1
        self.shortcut = None if not self.is_shortcut else self._shortcut(in_channels, out_channels, stride)

    def forward(self, X):
        out = self.conv1(X)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        # 当X的维度和out不一致时，需要用shortcut处理X
        out += X if not self.shortcut else self.shortcut(X)
        out = F.relu(out)
        return out

    def _shortcut(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )


class ResNet34(BasicModule):

    def __init__(self, num_classes=2):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),  # 64 * 112 * 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # 64 * 56 * 56
        )
        # layer1 不需要shortcut，因为图像没变化(kernel_size=3,stride=1, padding=1)
        self.layer1 = self._make_layer(64, 64, 3, 1)
        self.layer2 = self._make_layer(64, 128, 4, 2)
        self.layer3 = self._make_layer(128, 256, 6, 2)
        self.layer4 = self._make_layer(256, 512, 3, 2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, block_num, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(block_num - 1):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, X):
        # X: 3 * 224 * 224
        out = self.pre(X)  # 64 * 56 * 56
        out = self.layer1(out)  # 64 * 56 * 56
        out = self.layer2(out)  # 128 * 28 * 28
        out = self.layer3(out)  # 256 * 14 * 14
        out = self.layer4(out)  # 512 * 7 * 7
        out = F.avg_pool2d(out, 7)  # 512 * 1 * 1
        out = out.view(out.size(0), -1)  # 512
        out = self.fc(out)  # len(classification)
        return out
