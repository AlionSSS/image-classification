import zipfile
import os
from tqdm import tqdm


# 请先下载数据集 Dogs vs. Cats Redux: Kernels Edition
# 下载地址 https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data
# 保存到 ./dogs-vs-cats-redux-kernels-edition.zip

def download_extract_data():
    download_url = "https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data"
    data_path = "./dogs-vs-cats-redux-kernels-edition"
    data_zip_path = "./dogs-vs-cats-redux-kernels-edition.zip"
    data_train_zip_path = os.path.join(data_path, "train.zip")
    data_test_zip_path = os.path.join(data_path, "test.zip")

    if not os.path.exists(data_path):
        # 解压
        assert os.path.exists(data_zip_path), \
            f"项目下不存在数据文件 {data_zip_path} ，请先下载（{download_url}） 或 将该压缩文件放置到项目根目录下"
        with zipfile.ZipFile(data_zip_path) as zfile:
            for f in tqdm(zfile.infolist(), desc=f"Extracting... {data_zip_path}"):
                zfile.extract(f, data_path)

    if not os.path.exists(os.path.join(data_path, "train")):
        # 解压 train.zip
        with zipfile.ZipFile(data_train_zip_path) as zfile:
            for f in tqdm(zfile.infolist(), desc=f"Extracting... {data_train_zip_path}"):
                zfile.extract(f, data_path)

    if not os.path.exists(os.path.join(data_path, "test")):
        # 解压 test.zip
        with zipfile.ZipFile(data_test_zip_path) as zfile:
            for f in tqdm(zfile.infolist(), desc=f"Extracting... {data_test_zip_path}"):
                zfile.extract(f, data_path)


if __name__ == '__main__':
    download_extract_data()
