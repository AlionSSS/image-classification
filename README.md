# image-classification
This is an AI project about image classification.

## 数据集 Dogs vs. Cats Redux: Kernels Edition
1. 请先下载数据集 [dogs-vs-cats](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data) 
2. 将下载好的文件`dogs-vs-cats-redux-kernels-edition.zip`放置项目根目录
3. 执行项目根目录下的 [download_extract_data.py](download_extract_data.py)，完成解压

## 帮助信息
- 命令示例
```shell
python main.py help
```

## 模型训练
- 命令示例
```shell
python main.py train \
--model='ResNet34' \
--load-demo-path='checkpoints/resnet34_20230621_23:39:02.pth' \
--train-data-root=dogs-vs-cats-redux-kernels-edition/train \
--max-epoch=30 \
--batch-size=64 \
--lr=0.005
```

## 模型测试
- 命令示例
```shell
python main.py train \
--model='ResNet34' \
--load-demo-path='checkpoints/resnet34_20230621_23:39:02.pth' \
--test-data-root=dogs-vs-cats-redux-kernels-edition/test \
--batch-size=64
```