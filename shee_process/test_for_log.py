import re
import matplotlib.pyplot as plt


def parse_log(log_path):
    epochs = []
    losses = []
    lrs = []

    with open(log_path, 'r') as f:
        current_epoch = None
        current_loss = None
        current_lr = None

        for line in f:
            # 解析epoch和loss
            epoch_match = re.search(r'\[Epoch:\s+(\d+),\s+loss:\s+([-+]?\d+\.\d+)', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                current_loss = float(epoch_match.group(2))
                continue

            # 解析学习率
            lr_match = re.search(r'Cur lr:\s+([-+]?\d+\.\d+)', line)
            if lr_match:
                current_lr = float(lr_match.group(1))

                # 确保数据配对
                if current_epoch is not None:
                    epochs.append(current_epoch)
                    losses.append(current_loss)
                    lrs.append(current_lr)

                    # 重置临时变量
                    current_epoch = None
                    current_loss = None
                    current_lr = None

    return epochs, losses, lrs


def plot_training_curves(epochs, losses, lrs):
    plt.figure(figsize=(12, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, 'b-', linewidth=2)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 绘制学习率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, lrs, 'r-', linewidth=2)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    log_path = "./log_pretrain_bak.txt"  # 修改为你的log文件路径
    epochs, losses, lrs = parse_log(log_path)

    print(f"Parsed {len(epochs)} epochs of data")
    print(f"Loss range: [{min(losses):.3f}, {max(losses):.3f}]")
    print(f"Learning rate range: [{min(lrs):.5f}, {max(lrs):.5f}]")

    plot_training_curves(epochs, losses, lrs)
