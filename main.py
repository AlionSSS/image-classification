import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch import nn

import fire
import inspect
import pandas as pd
import tqdm

import models
from config import DefaultConfig, ExtDefaultConfig
from data import DogCatDataset
from utils import Visualizer

opt = ExtDefaultConfig()


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
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    previous_loss = 0.0
    best_val_acc = 0.0
    for epoch in range(1, opt.max_epoch + 1):
        epoch_flag = f"{epoch}/{opt.max_epoch}"
        model.train()
        train_loss = _train_epoch(epoch_flag, model, train_loader, criterion, optimizer, vis)

        model.eval()
        val_loss, val_accuracy = _val_epoch(epoch_flag, model, val_loader, criterion)

        vis.plot("epoch_train_loss", train_loss)
        vis.plot("epoch_val_loss", val_loss)
        vis.plot("epoch_val_acc", val_accuracy)
        vis.log(f"epoch: {epoch}/{opt.max_epoch}, lr: {lr:.6f}, "
                f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_accuracy:.4f}")

        # 只保存当前最佳accuracy的模型
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            model.save()

        # 如果损失变大，那就降低学习率
        if train_loss > previous_loss:
            lr *= opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = train_loss


def _train_epoch(epoch_flag: str, model: nn.Module, dataloader: DataLoader, criterion, optimizer: torch.optim.Optimizer,
                 vis: Visualizer) -> float:
    """
    训练-单个epoch
    """
    train_total_loss = 0.0
    progress = tqdm.tqdm(dataloader, desc=f"Train... [Epoch {epoch_flag}]")
    for i, (X, y) in enumerate(progress, 1):
        X, y = X.to(opt.device), y.to(opt.device)

        out = model(X)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_total_loss += loss.item()
        if i % opt.print_freq == 0:
            vis.plot("freq_loss", train_total_loss / i)
    train_avg_loss = train_total_loss / len(dataloader)
    return train_avg_loss


@torch.no_grad()
def _val_epoch(epoch_flag: str,model: nn.Module, dataloader: DataLoader, criterion) -> Tuple[float, float]:
    """
    验证-单个epoch
    """
    val_total_loss = 0.0
    val_total, val_correct = 0, 0
    progress = tqdm.tqdm(dataloader, desc=f"Valid... [Epoch {epoch_flag}]")
    for i, (X, y) in enumerate(progress, 1):
        X, y = X.to(opt.device), y.to(opt.device)

        out = model(X)
        loss = criterion(out, y)

        _, preds = torch.max(out, 1)
        val_total += y.size(0)
        val_correct += (preds == y).sum().item()

        val_total_loss += loss.item()
    val_avg_loss = val_total_loss / len(dataloader)
    val_accuracy = 100 * val_correct / val_total
    return val_avg_loss, val_accuracy


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

    if os.path.isfile(opt.result_file):
        os.remove(opt.result_file)

    model.eval()
    _test_epoch(model, test_loader)


@torch.no_grad()
def _test_epoch(model, dataloader):
    progress = tqdm.tqdm(dataloader, desc="Test ...")
    for i, (X, y) in enumerate(progress, 1):
        X = X.to(opt.device)
        out = model(X)
        _, preds = torch.max(out, 1)

        pd.DataFrame(data={
            'id': y,
            'label': preds.tolist()
        }).to_csv(opt.result_file, mode="a", header=i == 1, index=False)


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
