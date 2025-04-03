import torch
from config import load_args
from torch.utils.data.dataloader import DataLoader
from dataset import DownDataset
from model import Model, DownStreamModel
import torch.nn.functional as F


def get_confuse_matrix(pre_model_path='./checkpoints/epoch80_checkpoint_pretrain_model_bs128.pth',
                       model_path='./checkpoints/epoch160_down_model.pth',
                       test_path='./data/TFIs30_10/final/p2'):
    # 加载模型
    args = load_args()
    args.checkpoints = pre_model_path
    model = DownStreamModel(args)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    for param in model.parameters():  # 关梯度
        param.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 加载测试数据
    dataset = DownDataset(test_path, split_num=100)
    dataloader = DataLoader(dataset, batch_size=16)
    # test
    for data, label in dataloader:
        data, label = data.to(device), label.to(device)
        out = model(data)
        pre_label = F.softmax(out, dim=-1).max(-1)[1]
        print(pre_label, label)  # tensor([0], device='cuda:0') tensor([0], device='cuda:0')


if __name__ == '__main__':
    get_confuse_matrix()
