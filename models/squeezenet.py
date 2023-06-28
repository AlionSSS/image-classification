from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
from models.basic_module import BasicModule
from torch import nn
from torch.optim import Adam


class SqueezeNet(BasicModule):

    def __init__(self, num_classes=2):
        super(SqueezeNet, self).__init__()
        self.model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
        # 修改分类数 1000 -> 2
        self.model.num_classes = num_classes
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )

    def forward(self, x):
        return self.model(x)

    def get_optimizer(self, lr, weight_decay):
        # 有预训练模型，只训练后面部分
        return Adam(self.model.classifier.parameters(), lr, weight_decay=weight_decay)
