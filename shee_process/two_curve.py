import re
import matplotlib.pyplot as plt


def parse_log(log_path):
    epochs = []
    train_losses = []
    test_losses = []

    with open(log_path, 'r') as f:
        current_epoch = None
        loss_train = None
        loss_test = None

        for line in f:
            # 解析训练损失
            train_match = re.search(r'\[Down Task Train Epoch:\s+(\d+)\], loss: ([\d.]+), acc: ([\d.]+)', line)
            if train_match:
                current_epoch = int(train_match.group(1))
                loss_train = float(train_match.group(2))
                epochs.append(current_epoch)
                train_losses.append(loss_train)
                continue

            # 解析测试损失
            test_match = re.search(r'\[Down Task Test Epoch:\s+(\d+)\], loss: ([\d.]+), acc: ([\d.]+)', line)
            if test_match:
                loss_test = float(test_match.group(2))
                test_losses.append(loss_test)

    return epochs, train_losses, test_losses


def plot_combined_curves(data_list, labels, colors, linestyles):
    plt.figure(figsize=(12, 6))

    # 训练损失子图
    plt.subplot(1, 2, 1)
    for (epochs, train_losses, _), label, color, ls in zip(data_list, labels, colors, linestyles):
        plt.plot(epochs, train_losses, color=color, linestyle=ls, linewidth=2, label=label)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 测试损失子图
    plt.subplot(1, 2, 2)
    # for (_, test_losses, test_epochs), label, color, ls in zip(data_list, labels, colors, linestyles):
        # 注意测试损失可能没有对应的epoch记录，这里假设测试与训练epoch同步
        # plt.plot(test_epochs[:len(test_losses)], test_losses, color=color, linestyle=ls, linewidth=2, label=label)
    for (epochs, _, test_epochs), label, color, ls in zip(data_list, labels, colors, linestyles):
        plt.plot(epochs, test_epochs, color=color, linestyle=ls, linewidth=2, label=label)
    plt.title('Testing Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    # plt.savefig('combined_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 解析两个日志文件
    log_paths = ["./log_whole20250409.txt", "./log_whole_noProj20250410.txt"]
    all_data = [parse_log(path) for path in log_paths]
    # 设置曲线样式
    labels = ["Proj", "noProj"]
    colors = ["blue", "red"]
    linestyles = ["-", "--"]

    plot_combined_curves(all_data, labels, colors, linestyles)
