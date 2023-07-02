import warnings

import torch.cuda


class DefaultConfig(object):
    visdom_env = "image-classification"  # Visdom的环境
    model = "SqueezeNet"  # 使用的模型
    train_data_root = "dataset/dogs-vs-cats-redux-kernels-edition/train"  # 训练集的路径
    test_data_root = "dataset/dogs-vs-cats-redux-kernels-edition/test"  # 测试集的路径
    load_model_path = None  # 加载预训练模型的路径，None表示不加载

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用的设备，cpu或cuda
    print_freq = 10  # 打印信息间隔的step数

    debug_file = "/tmp/debug"
    result_file = "./results.csv"

    max_epoch = 10  # 训练的epoch数
    batch_size = 64  # 训练每批大小
    num_worker = 1  # 加载数据的worker进程数
    lr = 0.005  # 学习率
    lr_decay = 0.95  # 学习率衰减
    weight_decay = 1e-4

    num_classes = 2  # 分类数量


class ExtDefaultConfig(DefaultConfig):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn(f"Warning: opt has not attribute {k}")
            setattr(self, k, v)

    def print_attr(self):
        print("user config:")
        for k, v in self.__class__.__base__.__dict__.items():
            if not k.startswith("__"):
                print("\t" + k + " = " + str(getattr(self, k)))
