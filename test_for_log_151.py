import re
import matplotlib.pyplot as plt


def parse_log(log_path):
    epochs = []
    losses = []
    lrs = []

    with open(log_path, 'r') as f:
        current_epoch = None
        loss_train = None
        loss_test = None

        for line in f:
            epoch_match = re.search(r'\[Down Task Train Epoch:\s+(\d+)\], loss: ([\d.]+), acc: ([\d.]+)', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                loss_train = float(epoch_match.group(2))
                epochs.append(current_epoch)
                losses.append(loss_train)
                continue

            lr_match = re.search(r'\[Down Task Test Epoch:\s+(\d+)\], loss: ([\d.]+), acc: ([\d.]+)', line)
            if lr_match:
                loss_test = float(lr_match.group(2))
                lrs.append(loss_test)
            # 重置临时变量
            current_epoch = None
            loss_train = None
            loss_test = None

    return epochs, losses, lrs


def plot_training_curves(epochs, losses, lrs):
    plt.figure(figsize=(12, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, 'b-', linewidth=2)
    plt.title('Down Task Train Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 绘制学习率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, lrs, 'r-', linewidth=2)
    plt.title('Down Task Test Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('training_curves_1.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    log_path = "./log_151.txt"  # 修改为你的log文件路径
    epochs, losses, lrs = parse_log(log_path)

    print(f"Parsed {len(epochs)} epochs of data")
    print(f"Loss range: [{min(losses):.3f}, {max(losses):.3f}]")
    print(f"Learning rate range: [{min(lrs):.5f}, {max(lrs):.5f}]")

    plot_training_curves(epochs, losses, lrs)
