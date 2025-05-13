import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# 参数配置
FIG_SIZE = (8, 4)  # 图表尺寸（英寸）
FONT_SIZE = 10  # 字体大小
DPI = 300  # 输出分辨率
COLORS = ['#1f77b4', '#ff7f0e']  # 学术配色
LINE_WIDTH = 1.5  # 线宽
SAVE_FORMAT = 'png'  # 输出格式（pdf/png）


def calculate_differences(image_folder):
    """计算相邻图像之间的差异指标"""
    files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')],
                   key=lambda x: int(x.split('.')[0]))
    ssim_scores = []
    mse_scores = []

    for i in range(1, len(files)):
        img_prev = io.imread(os.path.join(image_folder, files[i - 1]))
        img_curr = io.imread(os.path.join(image_folder, files[i]))

        # 移除Alpha通道（保留前3个通道）
        img_prev = img_prev[..., :3]
        img_curr = img_curr[..., :3]

        # 转换为灰度
        img_prev = color.rgb2gray(img_prev)
        img_curr = color.rgb2gray(img_curr)

        # 计算指标
        ssim_val = ssim(img_prev, img_curr, data_range=1.0)
        mse_val = mean_squared_error(img_prev, img_curr)

        ssim_scores.append(ssim_val)
        mse_scores.append(mse_val)

    return np.array(ssim_scores), np.array(mse_scores)


def plot_difference_curves(ssim_scores, mse_scores, save_path):
    """绘制期刊级质量曲线图"""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'font.family': 'serif',
        'axes.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE - 2,
        'xtick.labelsize': FONT_SIZE - 2,
        'ytick.labelsize': FONT_SIZE - 2
    })

    fig, ax1 = plt.subplots(figsize=FIG_SIZE)

    # 绘制SSIM曲线
    x = np.arange(1, len(ssim_scores) + 1)
    ax1.plot(x, ssim_scores,
             color=COLORS[0],
             linewidth=LINE_WIDTH,
             marker='o',
             markersize=4,
             label='SSIM')

    ax1.set_xlabel('Iteration Step (n → n+1)')
    ax1.set_ylabel('Structural Similarity (SSIM)', color=COLORS[0])
    ax1.tick_params(axis='y', labelcolor=COLORS[0])
    ax1.set_ylim(0.8, 1.0)  # 根据实际数据调整

    # 创建第二个Y轴用于MSE
    ax2 = ax1.twinx()
    ax2.plot(x, mse_scores,
             color=COLORS[1],
             linewidth=LINE_WIDTH,
             linestyle='--',
             marker='s',
             markersize=4,
             label='MSE')

    ax2.set_ylabel('Mean Squared Error (MSE)', color=COLORS[1])
    ax2.tick_params(axis='y', labelcolor=COLORS[1])

    # 组合图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2,
               loc='upper center',
               frameon=False,
               ncol=2)

    # 网格和美化
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_xticks(np.arange(1, len(ssim_scores) + 1))
    ax1.set_xlim(0.5, len(ssim_scores) + 0.5)

    plt.title('Iterative Image Enhancement Evaluation', pad=15)
    plt.tight_layout()

    # 保存矢量图
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight', format=SAVE_FORMAT)
    plt.close()


if __name__ == "__main__":
    image_folder = "./temp1"  # 修改为你的图像文件夹路径
    output_path = "./test.png"  # 输出文件路径

    ssim_vals, mse_vals = calculate_differences(image_folder)
    plot_difference_curves(ssim_vals, mse_vals, output_path)
