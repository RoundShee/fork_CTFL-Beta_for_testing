import torch
from config import load_args
from torch.utils.data.dataloader import DataLoader
from dataset import DownDataset
from model import Model, DownStreamModel
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns


def get_confuse_matrix(pre_model_path='./checkpoints/epoch80_checkpoint_pretrain_model_bs128.pth',
                       model_path='./checkpoints/epoch100_down_model.pth',
                       test_path='./data/TFIs30_10/final/p2'):
    # 加载模型
    args = load_args()
    args.checkpoints = pre_model_path
    # 👇以下内容为args修改参数：
    # args.proj_hidden = 1024
    # args.proj_out = 512
    # args.pred_out = 512
    # 👆部分；
    model = DownStreamModel(args, n_classes=12)  # 👈注意这里-后续类别修改
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    for param in model.parameters():  # 关梯度
        param.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 加载测试数据
    dataset = DownDataset(test_path, split_num=200)
    dataloader = DataLoader(dataset, batch_size=20)
    all_real_label = []
    all_pre_label = []
    for data, label in dataloader:
        all_real_label.extend(label.tolist())
        data = data.to(device)
        out = model(data)
        pre_label = F.softmax(out, dim=-1).max(-1)[1]
        # print(pre_label, label)  # tensor([0], device='cuda:0') tensor([0], device='cuda:0')
        all_pre_label.extend(pre_label.tolist())
    matrix = confusion_matrix(all_real_label, all_pre_label)
    accuracy = accuracy_score(all_real_label, all_pre_label)
    nmi = normalized_mutual_info_score(all_real_label, all_pre_label)
    ari = adjusted_rand_score(all_real_label, all_pre_label)
    return matrix, accuracy, nmi, ari


if __name__ == '__main__':
    conf_matrix, accuracy, nmi, ari = get_confuse_matrix(
        pre_model_path='./checkpoints/epoch100_pretrain12_20250418.pth',
        model_path='./checkpoints/epoch180_down12_20250418.pth',
        test_path='./data/TFIs12_30_8/final/n10')
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.suptitle(f'SNR=-10dB Accuracy={accuracy:.2f}, NMI={nmi:.2f}, ARI={ari:.2f}')
    plt.savefig('./shee_process/0418_n10_radar12_down180.png', dpi=100, bbox_inches='tight')
    plt.show()
