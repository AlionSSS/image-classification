from .basic_module import BasicModule

from torch import nn
from torch.nn import functional as F


class LeNet(BasicModule):

    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, (5, 5))
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        # self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc1 = nn.Linear(53 * 53 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)  # 110 * 110
        out = F.max_pool2d(F.relu(self.conv2(out)), kernel_size=2)  # 53 * 53
        out = out.view(out.size()[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
