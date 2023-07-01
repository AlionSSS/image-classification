# image-classification
- This is an AI project about image-classification.
- models 
  - [x] ResNet34
  - [x] SqueezeNet
  - [x] LeNet

## Environment
- 主要使用 `PyTorch 1.12.1`、`Python 3.7.12`
- 依赖库详见 [requirements.txt](requirements.txt)

## Dataset 
- **Dogs vs. Cats Redux: Kernels Edition**
  1. 请先下载数据集 [dogs-vs-cats](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data) 
  2. 将下载好的文件`dogs-vs-cats-redux-kernels-edition.zip`放置项目根目录
  3. 执行项目根目录下的 [download_extract_data.py](download_extract_data.py)，完成解压


## Help Info
- 命令示例
```shell
python main.py help
```


## Model Train
- 先启动 Visdom Server，见 [Run Visdom](#run-visdom)
- 命令示例
```shell
# Linux
python main.py train \
--model='ResNet34' \
# --load-model-path='checkpoints/ResNet34_20230628_221458.pth' \
--train-data-root='dogs-vs-cats-redux-kernels-edition/train' \
--max-epoch=30 \
--batch-size=64 \
--lr=0.005
```

## Model Test
- 命令示例
```shell
# Linux
python main.py train \
--model='ResNet34' \
--load-model-path='checkpoints/ResNet34_20230628_221458.pth' \
--test-data-root='dogs-vs-cats-redux-kernels-edition/test' \
--batch-size=64
```

## Run Visdom
- 命令示例
```shell
# 阻塞启动
python -m visdom.server

# 非阻塞启动
nohup python -m visdom.server &
```
- 启动后即可使用Web浏览器访问 [http://localhost:8097](http://localhost:8097)
- 在网页选择环境`default`，通过图表查看训练过程中的`loss`、`accuracy`、`log`
![screenshot-2023-06-22 131844.png](resource%2Fscreenshot-2023-06-22%20131844.png)
