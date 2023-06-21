from .BasicModules import BasicModule

from torchvision.models import squeezenet1_0


class SqueezeNet(BasicModule):

    def __init__(self) -> None:
        super().__init__()
        self.model = squeezenet1_0()

    def __forward__(self, X):
        out = self.model(X)
        return out
