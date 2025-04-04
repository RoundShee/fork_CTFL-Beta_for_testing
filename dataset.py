from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
import torch
import os
import glob


class MyDataset(Dataset):
    def __init__(self, path):
        
        p_t1 = path + '/r1'
        p_t2 = path + '/r2'
        imgs1 = []
        imgs2 = []
        fh = os.listdir(p_t1)
        for line in fh:
            # line = line.rstrip()
            words = line.split()
            imgs1.append(p_t1 + "/" + words[0])
            imgs2.append(p_t2 + "/" + words[0])
            self.imgs1 = imgs1
            self.imgs2 = imgs2


    def __getitem__(self, index):
        fn1 = self.imgs1[index]
        fn2 = self.imgs2[index]
        #print(fn)
        t = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        img1 = Image.open(fn1).convert('RGB')
        img2 = Image.open(fn2).convert('RGB')
        img1 = t(img1)
        img2 = t(img2)
        return img1, img2


    def __len__(self):
        return len(self.imgs1)


class DownDataset(Dataset):
    """下采样测试训练dataloader"""
    def __init__(self, path, split_num=1000):
        """split是生成每种调制信号的数量"""
        self.split_num = split_num  # 后面俩函数可能要用
        self.file_list = sorted(glob.glob(os.path.join(path, "*.png")),
                                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        t = T.Compose(
            [T.Resize((224, 224)), T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        img1 = Image.open(img_path).convert('RGB')
        img1 = t(img1)
        label = idx // self.split_num
        return img1, label
