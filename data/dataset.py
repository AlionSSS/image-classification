import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision import transforms as T


class DogCatDataset(Dataset):
    """
    猫狗数据集

    划分为train、val、test，并做图像预处理
    """

    def __init__(self, root_path, mode=None) -> None:
        assert mode in {"train", "val", "test"}, "mode should in ('train','val','test')"

        self.mode = mode
        self.root_path = root_path
        img_url_all = os.listdir(root_path)

        if self.mode == "test":
            self.img_urls = img_url_all
        else:
            # 划分train和val数据集
            # 排序保证train和val的数据每次都不变
            img_url_all = sorted(img_url_all, key=lambda _: int(_.split(".")[-2]))
            ratio_idx = int(0.7 * len(img_url_all))
            self.img_urls = img_url_all[:ratio_idx] if self.mode == "train" else img_url_all[ratio_idx:]

        # 定义transforms
        if self.mode == "train":
            # train 增强，提高泛化能力
            self.transforms = T.Compose([
                T.Resize(256),
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            # test, val
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    def __getitem__(self, index) -> T_co:
        img_url = self.img_urls[index]
        if self.mode == "test":
            label = img_url.split(".")[-2]
        else:
            label = 1 if "dog" in img_url else 0

        data = Image.open(os.path.join(self.root_path, img_url))
        data = self.transforms(data)

        return data, label

    def __len__(self):
        return len(self.img_urls)
