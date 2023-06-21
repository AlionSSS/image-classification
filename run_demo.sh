# 打印帮助信息
python main.py help

# 训练模型
python main.py train \
--model='ResNet34' \
--load-demo-path='checkpoints/resnet34_20230621_23:39:02.pth' \
--train-data-root=dogs-vs-cats-redux-kernels-edition/train \
--max-epoch=30 \
--batch-size=64 \
--lr=0.005

# 测试模型
python main.py train \
--model='ResNet34' \
--load-demo-path='checkpoints/resnet34_20230621_23:39:02.pth' \
--test-data-root=dogs-vs-cats-redux-kernels-edition/test \
--batch-size=64