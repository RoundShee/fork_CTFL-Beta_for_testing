"""
本文档重新实现数据生成部分
"""

import numpy as np
import os
import glob
from signal_generate_roundshee import gen_one_chirp_sig, gen_one_bpsk, gen_one_2fsk, get_spwvd, gen_one_qpsk, gen_one_fre_sig, gen_one_vee_fre_sig
from scipy.io import savemat, loadmat
from copy_MSST import MSST_Y, SST, save_matlab_style_image
from concurrent.futures import ProcessPoolExecutor  # 多进程处理
from functools import partial  # 固定参数传入

# 全局采样频率Fs
Fs = 100e6  # 采样频率Fs=100MHz   故最大可分析载频为50MHz
Ts = 1 / Fs  # 两点实际间距-秒 0.01us=0.01e-6


def get_one_signal(which_one):
    sequence = np.array([])
    if which_one == 1:
        f_c = np.random.uniform(Fs/6, Fs/5)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        sequence = gen_one_bpsk(fs=Fs, carr_fre=f_c, pulse_width=pul_wid, code_speed=2e6, code_rand=False)
    elif which_one == 2:
        f_c = np.random.uniform(Fs/12, Fs/10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        sequence = gen_one_bpsk(fs=Fs, carr_fre=f_c, pulse_width=pul_wid, code_speed=2e6, code_rand=False)
    elif which_one == 3:
        f_c = np.random.uniform(Fs/12, Fs/10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        sequence = gen_one_qpsk(fs=Fs, carr_fre=f_c, pulse_width=pul_wid, code_speed=2e6, code_rand=False)
    elif which_one == 4:
        f_c = np.random.uniform(Fs / 4, Fs / 3)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        sequence = gen_one_qpsk(fs=Fs, carr_fre=f_c, pulse_width=pul_wid, code_speed=2e6, code_rand=False)
    elif which_one == 5:
        f_c = np.random.uniform(Fs/12, Fs/10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        f_delta = np.random.uniform(Fs/6, Fs/5) - f_c
        sequence = gen_one_2fsk(fs=Fs, f_c=f_c, f_delta=f_delta, pulse_width=pul_wid, code_speed=1.2e6, code_rand=False)
    elif which_one == 6:
        f_c = np.random.uniform(Fs / 12, Fs / 10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        f_delta = np.random.uniform(Fs / 6, Fs / 5) - f_c
        sequence = gen_one_2fsk(fs=Fs, f_c=f_c, f_delta=f_delta, pulse_width=pul_wid, code_speed=2e6, code_rand=False)
    elif which_one == 7:
        f_c = np.random.uniform(Fs / 6, Fs / 5)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        sequence = gen_one_fre_sig(fs=Fs, carr_fre=f_c, pulse_width=pul_wid)
    elif which_one == 8:
        f_c = np.random.uniform(Fs / 12, Fs / 10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        sequence = gen_one_fre_sig(fs=Fs, carr_fre=f_c, pulse_width=pul_wid)
    elif which_one == 9:
        f_c = np.random.uniform(Fs / 10, Fs / 8)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        chirp_rate = np.random.uniform(Fs / 6, Fs / 5) / pul_wid
        sequence = gen_one_chirp_sig(fs=Fs, carr_fre=f_c, chirp_rate=chirp_rate, pulse_width=pul_wid)
    elif which_one == 10:
        f_c = np.random.uniform(Fs / 12, Fs / 10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        chirp_rate = np.random.uniform(Fs / 12, Fs / 10) / pul_wid
        sequence = gen_one_chirp_sig(fs=Fs, carr_fre=f_c, chirp_rate=chirp_rate, pulse_width=pul_wid)
    elif which_one == 11:
        f_c = np.random.uniform(Fs / 6, Fs / 5)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        chirp_rate = np.random.uniform(Fs / 12, Fs / 10) / pul_wid  # 比例还与脉冲宽度有关
        sequence = gen_one_chirp_sig(fs=Fs, carr_fre=f_c, chirp_rate=-chirp_rate, pulse_width=pul_wid)
    elif which_one == 12:
        f_c = np.random.uniform(Fs / 6, Fs / 5)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        v_rate = np.random.uniform(Fs / 20, Fs / 15) / pul_wid * 2  # 比例还与脉冲宽度有关
        sequence = gen_one_vee_fre_sig(fs=Fs, carr_fre=f_c, v_rate=-v_rate, pulse_width=pul_wid)
    return sequence


def signal_raw_generate():
    snr_list = np.arange(-6, 16, 2)  # 信噪比备选表

    save_dir = './raw12/'  # 定义保存路径
    r1_dir = os.path.join(save_dir, 'r1')
    r2_dir = os.path.join(save_dir, 'r2')
    if not os.path.exists(r1_dir):
        os.makedirs(r1_dir)
    if not os.path.exists(r2_dir):
        os.makedirs(r2_dir)

    for j in range(0, 12):  # 雷达索引
        base_i = j*1000
        for i in range(base_i, base_i+500):  # 每种雷达只有500个
            sig = get_one_signal(j+1)
            snr_1, snr_2 = np.random.choice(snr_list, size=2, replace=False)
            sig1 = awgn(sig, snr_1)
            sig2 = awgn(sig, snr_2)
            file_name = '{:05d}.npy'.format(i)
            file_path1 = os.path.join(r1_dir, file_name)
            file_path2 = os.path.join(r2_dir, file_name)
            np.save(file_path1, sig1)
            np.save(file_path2, sig2)
    return 0


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

def try_spwvd():
    sig_raw = np.load('./raw/r1/5968.npy')
    sig_sp = get_spwvd(sig_raw, fs=Fs, window_length=128)
    save_matlab_style_image(sig_sp, 'test.png', target_size=(875, 656))


# try_spwvd()


def process_file(file_path, out_path, win_len, iter_num):
    """处理单个文件的独立函数-复用gen_TFIs重写成多进行处理"""
    try:
        sig_raw = np.load(file_path)
        ts, _ = MSST_Y(sig_raw, hlength=win_len, num=iter_num)

        filename = os.path.basename(file_path)
        output_name = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(out_path, output_name)

        save_matlab_style_image(ts, output_path, target_size=(875, 656))
        print(f"{file_path} OK")
        return True
    except Exception as e:
        print(f"{file_path} Failed: {str(e)}")
        return False


def gen_TFIs_with_CPUs(out_path='./TFIs30_10/r1', raw_path='./raw/r1', win_len=30, iter_num=10):
    """并行化主函数"""
    os.makedirs(out_path, exist_ok=True)
    npy_files = glob.glob(os.path.join(raw_path, '*.npy'))
    # 固定参数传递给子进程
    worker = partial(process_file, out_path=out_path, win_len=win_len, iter_num=iter_num)
    # 使用多进程池加速
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(worker, npy_files)
    # 统计成功/失败数量
    success = sum(results)
    print(f"Processed {len(npy_files)} files, {success} succeeded.")


def signal_raw_generate_final(out_path='./raw/final', num=200, noise='r'):
    """
    :param num: 不要超过1000否则覆盖
    :param out_path:
    :param noise: r:随机加噪, n:不加噪, num_int:指定信噪比的噪声
    :return:
    """
    snr_list = np.arange(-6, 16, 2)  # 信噪比备选表
    os.makedirs(out_path, exist_ok=True)
    for j in range(0, 12):
        base_i = j * 1000
        for i in range(base_i, base_i+num):
            sig = get_one_signal(j+1)
            if noise == 'r':
                snr_1 = np.random.choice(snr_list)  # get two diff snr
                sig = awgn(sig, snr_1)
            elif noise == 'n':
                # 模式2：不添加噪声
                pass  # 直接跳过加噪步骤
            elif isinstance(noise, (int, float)):
                # 模式3：指定信噪比加噪
                sig = awgn(sig, float(noise))
            else:
                raise ValueError("Invalid-noise-parameter.")
            file_name = '{:05d}.npy'.format(i)  # 生成文件名，格式为 0000.npy
            file_path1 = os.path.join(out_path, file_name)
            np.save(file_path1, sig)
    return 1


if __name__ == '__main__':  # md,多进程还怪麻烦的
    # signal_raw_generate()
    # gen_TFIs_with_CPUs(out_path='./TFIs12_30_8/r1', raw_path='./raw12/r1', iter_num=8)
    # gen_TFIs_with_CPUs(out_path='./TFIs12_30_8/r2', raw_path='./raw12/r2', iter_num=8)
    # pass
    signal_raw_generate_final(out_path='./raw12/final/n10', num=200, noise=-10)
    signal_raw_generate_final(out_path='./raw12/final/n6', num=200, noise=-6)
    signal_raw_generate_final(out_path='./raw12/final/n4', num=200, noise=-4)
    signal_raw_generate_final(out_path='./raw12/final/n2', num=200, noise=-2)
    signal_raw_generate_final(out_path='./raw12/final/n0', num=200, noise=0)
    signal_raw_generate_final(out_path='./raw12/final/p2', num=200, noise=2)
    gen_TFIs_with_CPUs(out_path='./TFIs12_30_8/final/n10', raw_path='./raw12/final/n10')
    gen_TFIs_with_CPUs(out_path='./TFIs12_30_8/final/n6', raw_path='./raw12/final/n6')
    gen_TFIs_with_CPUs(out_path='./TFIs12_30_8/final/n4', raw_path='./raw12/final/n4')
    gen_TFIs_with_CPUs(out_path='./TFIs12_30_8/final/n2', raw_path='./raw12/final/n2')
    gen_TFIs_with_CPUs(out_path='./TFIs12_30_8/final/n0', raw_path='./raw12/final/n0')
    gen_TFIs_with_CPUs(out_path='./TFIs12_30_8/final/p2', raw_path='./raw12/final/p2')
