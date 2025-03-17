"""
对MSST方法迁移到python
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# def MSST(sig_raw, win_len, iter_num):
#     sig_len, = sig_raw.shape
#     win_half_len = win_len//2
#     if win_len % 2 == 0:  # 保证窗长为奇数
#         win_len += 1
#     t = np.linspace(-0.5, 0.5, win_len)
#     gaussian_win = np.exp(-np.pi/0.32**2 * t**2)
#
#     trf = np.zeros((sig_len//2, sig_len), dtype=np.complex128)
#     for col_i in range(1, sig_len+1):
#         # tau = -min([round(N / 2) - 1, Lh, ti - 1]):min([round(N / 2) - 1, Lh, xrow - ti])
#         tau =


# 自己再写过于复杂,以下内容使用Deepseek实现：
def MSST_Y(x, hlength, num):
    x = x.reshape(-1)
    N = x.size
    hlength = hlength + 1 - (hlength % 2)
    ht = np.linspace(-0.5, 0.5, hlength)
    h = np.exp(-np.pi / (0.32**2) * ht**2)
    Lh = (hlength - 1) // 2

    round_N_over_2 = int(np.round(N / 2))
    tfr = np.zeros((N, N), dtype=np.complex128)

    for icol in range(N):
        ti = icol
        tau_min_neg = -min(round_N_over_2 - 1, Lh, ti)
        tau_max_pos = min(round_N_over_2 - 1, Lh, (N - 1) - ti)
        tau = np.arange(tau_min_neg, tau_max_pos + 1)
        if len(tau) == 0:
            continue
        valid_indices = ti + tau
        rSig = x[valid_indices]
        h_vals = h[Lh + tau].conj()
        indices = tau % N
        np.add.at(tfr, (indices, icol), rSig * h_vals)

    tfr = np.fft.fft(tfr, axis=0)
    tfr = tfr[:round_N_over_2, :]
    tfr1 = tfr.copy()

    omega = np.zeros((round_N_over_2, N), dtype=np.float64)
    for i in range(round_N_over_2):
        phase = np.unwrap(np.angle(tfr[i, :]))
        if len(phase) > 1:
            omega[i, :-1] = np.diff(phase) * N / (2 * np.pi)
            omega[i, -1] = omega[i, -2]
    omega = np.round(omega).astype(int)

    Ts = tfr.copy()
    for _ in range(num):
        Ts = SST(Ts, omega)

    Ts = Ts / (N / 2)
    return Ts, tfr1


def SST(tfr_f, omega_f):
    tfrm, tfrn = tfr_f.shape
    Ts_f = np.zeros_like(tfr_f, dtype=np.complex128)
    for b in range(tfrn):
        for eta in range(tfrm):
            k = omega_f[eta, b]
            if 0 <= k < tfrm:
                Ts_f[k, b] += tfr_f[eta, b]
    return Ts_f


def save_matlab_style_image(matrix, filename, target_size=(875, 656)):
    """
    仿MATLAB imagesc风格保存图像
    :param matrix: 输入的2D numpy数组 (200x400)
    :param filename: 保存路径（需包含.png扩展名）
    :param target_size: 目标像素尺寸 (width, height)
    """
    # 设置MATLAB风格的颜色映射（需安装对应的colormap）
    try:
        cmap = plt.colormaps['parula']  # MATLAB默认的parula colormap
    except KeyError:
        cmap = 'jet'  # 如果parula不可用则使用jet

    # 创建指定像素尺寸的figure
    dpi = 100  # 标准dpi
    fig_width = target_size[0] / dpi  # 转换为英寸
    fig_height = target_size[1] / dpi

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # 坐标轴占满整个figure

    # 显示图像（自动缩放数据范围）
    im = ax.imshow(np.abs(matrix),
                   cmap=cmap,
                   aspect='auto',  # 关闭宽高比自动调整
                   origin='upper',
                   interpolation='none',
                   norm=Normalize(vmin=np.min(np.abs(matrix)),
                                  vmax=np.max(np.abs(matrix))))

    # 隐藏坐标轴及边框
    ax.set_axis_off()

    # 保存图像（精确控制像素尺寸）
    plt.savefig(filename,
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0,
                facecolor='black')  # MATLAB默认背景色

    plt.close()
