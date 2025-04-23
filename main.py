import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from dataset import MyDataset, DownDataset

from config import load_args
from model import Model, DownStreamModel

import os
from datetime import datetime
import matplotlib.pyplot as plt

epochs = []  # 目前看来,这几句似乎也没用
acc_train = []
acc_test = []


def save_checkpoint(model, optimizer, args, epoch, down_model=False):
    print('\nModel Saving...')
    if args.device_num > 1:  # 这估计不是我能玩上的,下面是多GPU用容器封装使用的保存参数方法
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    if not down_model:
        torch.save({
            'model_state_dict': model_state_dict,
            'global_epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join('checkpoints', 'epoch'+str(epoch)+'_pretrain12_20250418.pth'))
    else:
        torch.save({
            'model_state_dict': model_state_dict,
            'global_epoch': epoch,
        }, os.path.join('checkpoints', 'epoch' + str(epoch) + '_down12_20250418.pth'))


def pre_train(epoch, train_loader, model, optimizer, args, f):
    model.train()  # 模型变更为训练模式

    losses, step = 0., 0.
    for x1, x2 in train_loader:  # 真狠,一次训练一轮
        if args.cuda:
            x1, x2 = x1.cuda(), x2.cuda()

        d1, d2 = model(x1, x2)
        optimizer.zero_grad()
        loss = d1 + d2
        loss.backward()
        optimizer.step()
        losses += loss.item()

        step += 1

    print('[Epoch: {0:4d}, loss: {1:.3f}'.format(epoch, losses / step), file = f)
    print('[Epoch: {0:4d}, loss: {1:.3f}'.format(epoch, losses / step))
    return losses / step


def _train(epoch, train_loader, model, optimizer, criterion, args, log):
    model.train()  # 对应下采样网络输出层训练模式
    losses, acc, step, total = 0., 0., 0., 0.
    for data, target in train_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        logits = model(data)

        optimizer.zero_grad()
        loss = criterion(logits, target)
        loss.backward()
        losses += loss.item()
        optimizer.step()

        pred = F.softmax(logits, dim=-1).max(-1)[1]
        acc += pred.eq(target).sum().item()

        step += 1
        total += target.size(0)
    print('[Down Task Train Epoch: {0:4d}], loss: {1:.3f}, acc: {2:.3f}'.format(epoch, losses / step, acc / total * 100.), file=log)
    print('[Down Task Train Epoch: {0:4d}], loss: {1:.3f}, acc: {2:.3f}'.format(epoch, losses / step, acc / total * 100.))
    log.flush()

def _eval(epoch, test_loader, model, criterion, args,log):
    model.eval()  # 评估模式-应该是下采样器要训练完以后在测试集上进行实时的评估
    losses, acc, step, total = 0., 0., 0., 0.
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            logits = model(data)
            loss = criterion(logits, target)
            losses += loss.item()
            pred = F.softmax(logits, dim=-1).max(-1)[1]
            acc += pred.eq(target).sum().item()

            step += 1
            total += target.size(0)
        print('[Down Task Test Epoch: {0:4d}], loss: {1:.3f}, acc: {2:.3f}'.format(epoch, losses / step, acc / total * 100.),file=log)
        print('[Down Task Test Epoch: {0:4d}], loss: {1:.3f}, acc: {2:.3f}'.format(epoch, losses / step, acc / total * 100.))
        log.flush()


def train_eval_down_task(down_model, down_train_loader, down_test_loader, args, log):
    global epochs  # 这里用到了全局变量,不是和特征提取器一致了
    down_optimizer = optim.SGD(down_model.parameters(), lr=args.down_lr, weight_decay=args.weight_decay, momentum=args.momentum)
    down_criterion = nn.CrossEntropyLoss()  # 下采样的交叉熵损失函数
    down_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(down_optimizer, T_max=args.down_epochs)
    for epoch in range(1, args.down_epochs + 1):
        epochs.append(epoch)
        _train(epoch, down_train_loader, down_model, down_optimizer, down_criterion, args, log)
        _eval(epoch, down_test_loader, down_model, down_criterion, args, log)
        down_lr_scheduler.step()
        print('Cur lr: {0:.5f}'.format(down_lr_scheduler.get_last_lr()[0]), file = log)  # 记录当前学习率
        log.flush()
        if epoch % args.print_intervals == 0:
            save_checkpoint(model=down_model, args=args, epoch=epoch, down_model=True, optimizer=None)


def main(args):
    # args.checkpoints = './checkpoints/checkpoint_pretrain_model.pth'  # 这句话没用上,传参再传
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    log = open('shee_process/log.txt', 'a+')

    if args.pretrain:
        if args.checkpoints:
            pass  # 这里是计划能够接续训练,现在用处不大
        model = Model(args)  # 这里是首次训练加载模型,先这么改
        if args.cuda:
            model = model.cuda()
        data = MyDataset("./data/TFIs12_30_8")
        trainloader = DataLoader(data, batch_size=16, shuffle=True, drop_last=True, num_workers=4)
        print("pretrain", file = log)
        log.flush()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800)  # 学习率调度器,针对优化器进行修改,动态调整学习率
        for epoch in range(1, args.epochs + 1):
            train_loss = pre_train(epoch, trainloader, model, optimizer, args, log)
            if epoch % args.print_intervals == 0:
                save_checkpoint(model, optimizer, args, epoch)  # 检查点保存
            lr_scheduler.step()  # 更新学习率
            print('Cur lr: {0:.5f}'.format(lr_scheduler.get_last_lr()[0]), file = log)  # 记录当前学习率
            print('Cur lr: {0:.5f}'.format(lr_scheduler.get_last_lr()[0]))
            log.flush()
    if args.train_down:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] train_down", file=log)
        log.flush()
        down_model = DownStreamModel(args, n_classes=12)  # 下采样模型,直接包含上面特征提取网络,注意args传入checkpoint
        if args.cuda:
            down_model = down_model.cuda()
        # 训练集:又生成了一部分没有加噪的数据作为训练集, 评估集就用r1
        train_data = DownDataset('./data/TFIs12_30_8/r1', split_num=500)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True, num_workers=4)
        test_data = DownDataset('./data/TFIs12_30_8/r2',  split_num=500)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=True, drop_last=True, num_workers=4)
        train_eval_down_task(down_model=down_model, down_train_loader=train_loader, down_test_loader=test_loader,
                             args=args, log=log)


if __name__ == '__main__':
    args = load_args()
    main(args)
