import time

import torch
from torch import nn
import os

class BasicModule(nn.Module):
    """
    基础Module，封装加载模型和保存模型的逻辑
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_name = type(self).__name__

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if not name:
            name = time.strftime(self.model_name + "_%Y%m%d_%H%M%S.pth")
        torch.save(self.state_dict(), os.path.join("checkpoints", name))
