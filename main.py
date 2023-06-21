import torch.optim
from torch.utils.data import DataLoader
from torch import nn

import fire
import inspect

import models
from config import DefaultConfig, ExtDefaultConfig
from data import DogCatDataset
from utils import Visualizer

opt = ExtDefaultConfig()
# new_config = {'lr': 0.005, 'use_gpu': False, "xxx": 999}
# opt.parse(**new_config)



#
# lr = opt.lr
#
# train_dataset = DogCatDataset(root_path=opt.train_data_root, mode="train")
#
# model = getattr(models, 'SqueezeNet')()
# # model = eval("models.SqueezeNet")()
# model.printxxx()

def train(**kwargs):
    """
    训练
    :param kwargs:
    :return:
    """

    opt.parse(**kwargs)
    opt.print_attr()

    vis = Visualizer(opt.visdom_env)

    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model = model.to(opt.device)

    train_data = DogCatDataset(root_path=opt.train_data_root, mode="train")
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_worker)
    val_data = DogCatDataset(root_path=opt.train_data_root, mode="val")
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_worker)

    lr = opt.lr
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    previous_loss = 0.0

    for epoch in range(1, opt.max_epoch):
        model.train()
        train_total_loss = 0.0
        train_avg_loss = 0.0
        for i, (X, y) in enumerate(train_loader, 1):
            X, y = X.to(opt.device), y.to(opt.device)

            out = model(X)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_total_loss += loss.item()
            if i % opt.print_freq == 0:
                train_avg_loss = train_total_loss / i
                vis.plot("loss", train_avg_loss)

        model.save()

        val_acc = val(model, val_loader, criterion)
        vis.plot("val_acc", val_acc)
        vis.log(f"epoch:{epoch}, lr:{opt.lr}, loss:{train_avg_loss}")

        # 如果损失变大，那就降低学习率
        if train_avg_loss > previous_loss:
            lr *= opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = train_avg_loss


def val(model: nn.Module, dataloader: DataLoader, criterion) -> float:
    """
    验证
    """
    model.eval()
    valid_total_loss = 0.0
    total, correct = 0, 0
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader, 1):
            X, y = X.to(opt.device), y.to(opt.device)

            out = model(X)
            loss = criterion(out, y)

            _, preds = torch.max(out, 1)
            total += y.size(0)
            correct += (preds == y).sum()

            valid_total_loss += loss.item()
            if i % 10 == 0 or i == len(dataloader):
                accurrcy = 100 * correct.item() / total
    return accurrcy


def test(**kwargs):
    """
    测试
    :param kwargs:
    :return:
    """

    opt.parse(**kwargs)
    opt.print_attr()

    test_data = DogCatDataset(root_path=opt.test_data_root, mode="test")
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_worker)

    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    results = []
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader, 1):
            X, y = X.to(opt.device), y.to(opt.device)

            out = model(X)
            _, preds = torch.max(out, 1)
            for xxx in zip(y, preds):
                print(xxx)


def help():
    """
    打印帮助信息
    :return:
    """
    print("""
    usage: python {0} <function> [--args=value,]
    <function> := train | test | help
    examples:
            python {0} train --visdom_env='env123' --lr=0.01
            python {0} test --test-data-root='dogs-vs-cats-redux-kernels-edition/test'
            python {0} help
    avaiable args:
    """.format("main.py"))

    source = inspect.getsource(DefaultConfig)
    print(source)


if __name__ == '__main__':
    fire.Fire()
