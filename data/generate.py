"""
本文档重新实现数据生成部分
"""

import numpy as np
import os
import glob
from signal_generate_roundshee import gen_one_chirp_sig, gen_one_bpsk, gen_one_2fsk
from scipy.io import savemat, loadmat
from copy_MSST import MSST_Y, SST, save_matlab_style_image


# 全局采样频率Fs
Fs = 100e6  # 采样频率Fs=100MHz   故最大可分析载频为50MHz
Ts = 1 / Fs  # 两点实际间距-秒 0.01us=0.01e-6


def signal_raw_generate():
    """
    生成原始数据,不做处理
    :return:
    """
    t = np.arange(0, 400*Ts, Ts)  # 脉冲宽度固定,400个采样点
    snr_list = np.arange(-6, 16, 2)  # 信噪比备选表

    save_dir = './raw/'  # 定义保存路径
    r1_dir = os.path.join(save_dir, 'r1')
    r2_dir = os.path.join(save_dir, 'r2')
    if not os.path.exists(r1_dir):
        os.makedirs(r1_dir)
    if not os.path.exists(r2_dir):
        os.makedirs(r2_dir)

    # CW 载频45MHz 变化范围:1/4到1/2
    for i in range(1000):  # should I keep them one by one ?
        fc = 45e6 * np.random.uniform(0.25, 0.5)  # 当前载频
        sig = np.cos(2 * np.pi * fc * t + np.random.uniform(0, 2 * np.pi))  # raw signal
        snr_1, snr_2 = np.random.choice(snr_list, size=2, replace=False)  # get two diff snr
        sig1 = awgn(sig, snr_1)
        sig2 = awgn(sig, snr_2)
        file_name = '{:04d}.npy'.format(i)  # 生成文件名，格式为 0000.npy
        file_path1 = os.path.join(r1_dir, file_name)
        file_path2 = os.path.join(r2_dir, file_name)
        np.save(file_path1, sig1)  # 保存数据
        np.save(file_path2, sig2)  # 保存数据

    # LFW
    for i in range(1000, 2000):
        fc = np.random.uniform(Fs / 10, Fs / 8)
        chirp = np.random.uniform(Fs / 6, Fs / 5) / (400*Ts)  # chirp完它频率不超出-直接抄radar9
        sig = gen_one_chirp_sig(fs=Fs, carr_fre=fc, chirp_rate=chirp, pulse_width=400*Ts)
        snr_1, snr_2 = np.random.choice(snr_list, size=2, replace=False)  # get two diff snr
        sig1 = awgn(sig, snr_1)
        sig2 = awgn(sig, snr_2)
        file_name = '{:04d}.npy'.format(i)
        file_path1 = os.path.join(r1_dir, file_name)
        file_path2 = os.path.join(r2_dir, file_name)
        np.save(file_path1, sig1)
        np.save(file_path2, sig2)

    # BPSK  现在来看有很大的问题,上面的chirp不对,要随机,还有选出的俩SNR必须要不一样,这个必须规避
    for i in range(2000, 3000):
        fc = np.random.uniform(Fs / 6, Fs / 5)
        sig = gen_one_bpsk(fs=Fs, carr_fre=fc, pulse_width=400*Ts, code_speed=2e6)
        snr_1, snr_2 = np.random.choice(snr_list, size=2, replace=False)
        sig1 = awgn(sig, snr_1)
        sig2 = awgn(sig, snr_2)
        file_name = '{:04d}.npy'.format(i)
        file_path1 = os.path.join(r1_dir, file_name)
        file_path2 = os.path.join(r2_dir, file_name)
        np.save(file_path1, sig1)
        np.save(file_path2, sig2)

    # 2FSK
    for i in range(3000, 4000):
        fc = np.random.uniform(25e6, 30e6)
        fc_delta = np.random.uniform(5e6, 10e6)
        sig = gen_one_2fsk(fs=Fs, f_c=fc, f_delta=fc_delta, pulse_width=400*Ts, code_speed=2e6)
        snr_1, snr_2 = np.random.choice(snr_list, size=2, replace=False)
        sig1 = awgn(sig, snr_1)
        sig2 = awgn(sig, snr_2)
        file_name = '{:04d}.npy'.format(i)
        file_path1 = os.path.join(r1_dir, file_name)
        file_path2 = os.path.join(r2_dir, file_name)
        np.save(file_path1, sig1)
        np.save(file_path2, sig2)

    # NLFM 非线性频率调制,根据其原代码的描述,相位对t求导,得$2\pi f_c $
    for i in range(4000, 5000):
        fc = np.random.uniform(25e6, 30e6)
        sig = np.cos(2 * np.pi * fc * t - 2 * np.pi*np.random.uniform(6, 8)*np.cos(2e6*t + np.random.uniform(0, 2 * np.pi)))
        snr_1, snr_2 = np.random.choice(snr_list, size=2, replace=False)
        sig1 = awgn(sig, snr_1)
        sig2 = awgn(sig, snr_2)
        file_name = '{:04d}.npy'.format(i)
        file_path1 = os.path.join(r1_dir, file_name)
        file_path2 = os.path.join(r2_dir, file_name)
        np.save(file_path1, sig1)
        np.save(file_path2, sig2)

    # LFM/NLFM
    for i in range(5000, 6000):
        fc = np.random.uniform(25e6, 30e6)
        sig = np.cos(2 * np.pi * fc * t - 2 * np.pi*np.random.uniform(6, 8)*np.cos(2e6*t + np.random.uniform(0, 2 * np.pi)))
        fc = np.random.uniform(Fs / 10, Fs / 8)
        chirp = np.random.uniform(Fs / 6, Fs / 5) / (400 * Ts)
        sig = sig + gen_one_chirp_sig(fs=Fs, carr_fre=fc, chirp_rate=chirp, pulse_width=400 * Ts)
        snr_1, snr_2 = np.random.choice(snr_list, size=2, replace=False)
        sig1 = awgn(sig, snr_1)
        sig2 = awgn(sig, snr_2)
        file_name = '{:04d}.npy'.format(i)
        file_path1 = os.path.join(r1_dir, file_name)
        file_path2 = os.path.join(r2_dir, file_name)
        np.save(file_path1, sig1)
        np.save(file_path2, sig2)

    return t


def awgn(sig, p1):
    """
    给信号 Sig 添加指定信噪比 p1（dB）的高斯白噪声-用AI生成的
    :param sig: 原始信号
    :param p1: 信噪比（dB）
    :return: 加噪后的信号
    """
    # 计算信号的功率
    signal_power = np.mean(np.abs(sig) ** 2)
    # 将信噪比从 dB 转换为线性比例
    snr_linear = 10 ** (p1 / 10)
    # 计算噪声的功率
    noise_power = signal_power / snr_linear
    # 计算噪声的标准差
    noise_std = np.sqrt(noise_power)
    # 生成高斯白噪声
    noise = np.random.normal(0, noise_std, sig.shape)
    # 将噪声添加到信号上
    sig1 = sig + noise
    return sig1


# t = signal_raw_generate()
# loaded_arr = np.load('raw/r1/0000.npy')
# print(loaded_arr.shape)


# 生成初步复现测试模型的时频图集
def gen_TFIs(out_path='./TFIs30_10/r1', raw_path='./raw/r1', win_len=30, iter_num=10):
    os.makedirs(out_path, exist_ok=True)
    npy_files = glob.glob(os.path.join(raw_path, '*.npy'))
    for file_path in npy_files:
        sig_raw = np.load(file_path)
        ts, _ = MSST_Y(sig_raw, hlength=win_len, num=iter_num)

        filename = os.path.basename(file_path)
        output_name = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(out_path, output_name)

        save_matlab_style_image(ts, output_path, target_size=(875, 656))

        print(file_path+' OK')


# gen_TFIs(out_path='./TFIs30_10/r1', raw_path='./raw/r1')
# gen_TFIs(out_path='./TFIs30_10/r2', raw_path='./raw/r2')
