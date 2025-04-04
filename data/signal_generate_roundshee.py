"""
review:
这文档本来是我自己尝试复现一篇名为:小_零样本雷达信号智能分选与未知信号识别方法研究  的论文所写的数据生成代码
现调用其中的信号生成部分.
以下是原始内容:

本文档首先专注于成电小样本部分生成

"""
import torch
import os
import numpy as np
import scipy.io as sio
from scipy.signal import windows
import matplotlib.pyplot as plt
import numba as nb

# 全局采样频率Fs
Fs = 100e6  # 采样频率Fs=100MHz   故最大可分析载频为50MHz
Ts = 1 / Fs  # 两点实际间距-秒 0.01us=0.01e-6


# 以下内容基于上述采样率进行
@nb.jit(nopython=True, parallel=True)
def spwvd(signal, fs, g, h):
    """
    编译加速计算
    """
    length_signal = len(signal)
    t = np.arange(length_signal)
    f = np.linspace(0, fs / 2, length_signal // 2 + 1)
    M = len(t)
    K = len(f)
    spwvd_result = np.zeros((K, length_signal), dtype=np.complex128)

    window_length = len(g)

    for m in nb.prange(M):
        tau_max = min(window_length // 2, length_signal - t[m] - 1, t[m])
        tau = np.arange(-tau_max, tau_max + 1)
        theta_max = min(window_length // 2, length_signal - t[m] - 1, t[m])
        theta = np.arange(-theta_max, theta_max + 1)
        for k in range(K):
            sum_val = 0
            for th in theta:
                for ta in tau:
                    idx1 = int(t[m] + ta / 2 - th)
                    idx2 = int(t[m] - ta / 2 - th)
                    if 0 <= idx1 < length_signal and 0 <= idx2 < length_signal:
                        h_index = min(th + theta_max, window_length - 1)
                        g_index = min(ta + tau_max, window_length - 1)
                        sum_val += h[h_index] * g[g_index] * \
                                   signal[idx1] * np.conj(signal[idx2]) * \
                                   np.exp(-1j * 2 * np.pi * f[k] * ta / fs)
            spwvd_result[k, m] = sum_val

    return spwvd_result


def gen_fixedPRI(CF=8e6, PW=1e-6, PRI=100e-6, duration=1e-3):
    """
    生成固定PRI脉冲信号--单频信号
    :param CF: 固定载波频率
    :param PW: 固定脉冲宽度
    :param PRI: 固定脉冲重复间隔
    :param duration: 指定输出序列的长度，单位为秒
    :return: 指定长度的采样序列 np格式
    """
    # 计算每个脉冲所需的采样点数
    samples_per_pulse = int(PW * Fs)
    # 计算每个PRI所需的采样点数
    samples_per_PRI = int(PRI * Fs)
    # 计算指定时长内的总采样点数
    total_samples = int(duration * Fs)
    # 初始化脉冲序列
    pulse_sequence = np.zeros(total_samples)
    # 生成脉冲
    pulse_index = 0
    while True:
        start_index = pulse_index * samples_per_PRI
        if start_index >= total_samples:
            break
        end_index = start_index + samples_per_pulse
        if end_index > total_samples:
            end_index = total_samples
        t = np.arange(start_index, end_index) * Ts
        # 相位相参计算出幅度
        pulse = np.cos(2 * np.pi * CF * t)
        pulse_sequence[start_index:end_index] = pulse
        pulse_index += 1
    return pulse_sequence
# 显然，gen_fixedPRI不合适


def get_spwvd(signal, fs, window_length=128):
    """
    通过调用SPWVD函数得到时频图矩阵
    :param signal: 输入信号numpy
    :param fs: 信号采样率
    :param window_length: 窗长度
    :return: 返回时频矩阵numpy
    """
    # windows.gaussian(N, std=N/6) 高斯窗
    # windows.blackman(N) 布莱克曼窗
    # windows.hann(window_length) 汉宁
    g = np.array(windows.hann(window_length))
    h = np.array(windows.hann(window_length))
    out_spwvd = spwvd(signal, fs, g, h)
    return out_spwvd


def plot_spwvd(spwvd_matrix, fs):
    """
    绘制平滑伪魏格纳 - 威利分布（SPWVD）的时频图
    :param spwvd_matrix: 时频矩阵
    :param fs: 采样率
    """
    # 创建一个新的图形窗口
    plt.figure(figsize=(10, 6))
    # 绘制时频图
    plt.imshow(np.abs(spwvd_matrix), aspect='auto', origin='lower',
               extent=[0, len(spwvd_matrix[0]) / fs * 1e6, 0, fs / 2 / 1e6])
    # 添加颜色条
    plt.colorbar(label='Magnitude')
    # 设置 x 轴标签，单位为 μs
    plt.xlabel('Time [μs]')
    # 设置 y 轴标签，单位为 MHz
    plt.ylabel('Frequency [MHz]')
    # 设置图的标题
    plt.title('Smoothing Pseudo Wigner - Ville Distribution')
    # 显示图形
    plt.show()


def gen_one_fre_sig(fs, carr_fre, pulse_width, random_phase=1):
    """
    生成单频信号一次脉冲
    :param fs:采样率
    :param carr_fre:载频
    :param pulse_width:脉冲宽度
    :param random_phase: 随机初始相位
    :return: 输出脉冲numpy序列
    """
    output_len = int(fs * pulse_width)  # 输出长度
    output_time = np.arange(output_len) / fs  # 生成时间序列
    if random_phase:
        phi_0 = np.random.uniform(0, 2 * np.pi)
    else:
        phi_0 = 0
    output = np.cos(2 * np.pi * carr_fre * output_time + phi_0)
    return output


def gen_one_chirp_sig(fs, carr_fre, chirp_rate, pulse_width, random_phase=1):
    """
    生成线性调频一次脉冲信号
    :param fs: 采样频率
    :param carr_fre: 载波频率
    :param chirp_rate: 调制斜率，单位为 Hz/s
    :param pulse_width: 脉宽
    :param random_phase: 随机初始相位，1 表示使用随机相位，0 表示初始相位为 0
    :return: numpy类型序列
    """
    num_samples = int(fs * pulse_width)  # 计算采样点数
    output_time = np.arange(num_samples) / fs  # 生成时间序列
    if random_phase:
        phi = np.random.uniform(0, 2 * np.pi)
    else:
        phi = 0
    chirp_signal = np.cos(2 * np.pi * carr_fre * output_time + np.pi * chirp_rate * output_time ** 2 + phi)
    return chirp_signal


def gen_one_vee_fre_sig(fs, carr_fre, v_rate, pulse_width, random_phase=1):
    """
    生成V型调制信号一个脉冲
    :param fs: 采样率
    :param carr_fre: 载波，初始频率
    :param v_rate: 调制频率 Hz/s
    :param pulse_width: 脉冲宽度
    :param random_phase: 随机初始相位，1 表示使用随机相位，0 表示初始相位为 0
    :return: numpy类型序列
    """
    num_samples = int(fs * pulse_width)  # 计算采样点数
    output_time = np.arange(num_samples) / fs  # 生成时间序列
    if random_phase:
        phi_0 = np.random.uniform(0, 2 * np.pi)
    else:
        phi_0 = 0
    vee_fre_sig = np.zeros(num_samples)  # 初始化信号数组
    half_pulse_width = int(num_samples/2)  # 脉冲持续时间的一半
    phi_1 = (output_time[0:half_pulse_width]*2*np.pi*carr_fre +
             np.pi*v_rate*output_time[0:half_pulse_width]**2)
    phi_2 = (2*np.pi*(carr_fre+v_rate*pulse_width)*output_time[half_pulse_width:] -
             np.pi*v_rate*output_time[half_pulse_width:]**2 -
             np.pi*v_rate*pulse_width**2 / 2)
    vee_fre_sig[0:half_pulse_width] = np.cos(phi_1 + phi_0)
    vee_fre_sig[half_pulse_width:] = np.cos(phi_2 + phi_0)
    return vee_fre_sig


def gen_one_bpsk(fs, carr_fre, pulse_width, code_speed, random_phase=1):
    """
    根据成电论文，生成指定参数下，模拟接收到的中频处理后的信号，处理后载波为2倍输入值。但根据论文提供的时频图以及测试参数，
    这里的bpsk为直接接收结果，不考虑2倍处理。 码信息随机生成
    :param fs:采样率-Hz
    :param carr_fre:载频-Hz
    :param pulse_width:脉宽-s
    :param code_speed:码元速率-bps
    :param random_phase:
    :return:输出仍是ndarray形式
    """
    output_len = int(fs * pulse_width)  # 输出长度
    output_time = np.arange(output_len) / fs  # 生成时间序列
    if random_phase:
        phi_0 = np.random.uniform(0, 2 * np.pi)
    else:
        phi_0 = 0
    code_fs_num = int(fs/code_speed)  # 一个码有几个点
    one_pulse_code_nums = int(output_len // code_fs_num + 1)  # 需要大概的码数量
    codes = np.random.randint(0, 2, size=one_pulse_code_nums)  # 生成随机码序列 BPSK是randint(0,2)不含2
    codes = np.repeat(codes, code_fs_num)  # 生成码序列对应脉冲相位
    codes = codes[:output_len]  # 截取与脉冲长度对应
    out_phase = 2*np.pi*carr_fre*output_time + phi_0 + np.pi*codes  # 相位计算
    out_bpsk = np.cos(out_phase)
    return out_bpsk


def gen_one_qpsk(fs, carr_fre, pulse_width, code_speed, random_phase=1):
    """
    如上QPSK
    :param fs:采样率-Hz
    :param carr_fre:载频-Hz
    :param pulse_width:脉宽-s
    :param code_speed:码元速率-bps
    :param random_phase:
    :return:输出仍是ndarray形式
    """
    output_len = int(fs * pulse_width)  # 输出长度
    output_time = np.arange(output_len) / fs  # 生成时间序列
    if random_phase:
        phi_0 = np.random.uniform(0, 2 * np.pi)
    else:
        phi_0 = 0
    code_fs_num = int(fs/code_speed)  # 一个码有几个点
    one_pulse_code_nums = int(output_len // code_fs_num + 1)  # 需要大概的码数量
    codes = np.random.randint(0, 4, size=one_pulse_code_nums)  # 生成随机码序列
    codes = np.repeat(codes, code_fs_num)  # 生成码序列对应脉冲相位
    codes = codes[:output_len]  # 截取与脉冲长度对应
    out_phase = 2*np.pi*carr_fre*output_time + phi_0 + np.pi*(codes/2+0.25)  # 相位计算
    out_qpsk = np.cos(out_phase)
    return out_qpsk


def gen_one_2fsk(fs, f_c, f_delta, pulse_width, code_speed, random_phase=1):
    """
    2FSK生成 频率表达式：f(t)=f_c+f_delta*code(tau)
    :param fs: 采样率
    :param f_c:
    :param f_delta:
    :param pulse_width:
    :param code_speed:
    :param random_phase:
    :return:
    """
    output_len = int(fs * pulse_width)  # 输出长度
    output_time = np.arange(output_len) / fs  # 生成时间序列
    if random_phase:
        phi_0 = np.random.uniform(0, 2 * np.pi)
    else:
        phi_0 = 0
    code_fs_num = int(fs / code_speed)  # 一个码有几个点
    one_pulse_code_nums = int(output_len // code_fs_num + 1)  # 需要大概的码数量
    codes = np.random.randint(0, 2, size=one_pulse_code_nums)  # 生成随机码序列 BPSK是randint(0,2)不含2
    codes = np.repeat(codes, code_fs_num)  # 生成码序列对应脉冲相位
    codes = codes[:output_len]  # 截取与脉冲长度对应
    out_phase = 2*np.pi*(f_c+f_delta*codes)*output_time+phi_0
    out_2fsk = np.cos(out_phase)
    return out_2fsk


# sio.savemat('MATLAB/pulse_sequence.mat', {'pulse_sequence': pulse_sequence})
# pulse_sequence = gen_fixedPRI(CF=25e6, PW=4e-6, PRI=10e-6, duration=4e-6)  # 测试代码
# pulse_sequence = gen_one_fre_sig(fs=Fs, carr_fre=25e6, pulse_width=4e-6)  # 单频信号
# pulse_sequence = gen_one_chirp_sig(fs=Fs, carr_fre=25e6, chirp_rate=2e12, pulse_width=4e-6)  # 线性调频
# pulse_sequence = gen_one_vee_fre_sig(fs=Fs, carr_fre=25e6, v_rate=-6e12, pulse_width=4e-6)  # V调频
# pulse_sequence = gen_one_bpsk(fs=Fs, carr_fre=25e6, pulse_width=4e-6, code_speed=2e6)  # bpsk信号  与单频相比不明显
# pulse_sequence = gen_one_qpsk(fs=Fs, carr_fre=25e6, pulse_width=4e-6, code_speed=2e6)  # qpsk信号
# pulse_sequence = gen_one_2fsk(fs=Fs, f_c=20e6, f_delta=10e6, pulse_width=4e-6, code_speed=2e6)
# spwvd_matrix = get_spwvd(pulse_sequence, Fs, window_length=128)
# plot_spwvd(spwvd_matrix, Fs)


def create_12_radar():
    """
    根据论文表2-1产生数据集
    """
    # radar1
    save_dir = './resource/small_sample_data_chapter2/train/radar01'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i1 in range(0, 80):
        f_c = np.random.uniform(Fs/6, Fs/5)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        sequence = gen_one_bpsk(fs=Fs, carr_fre=f_c, pulse_width=pul_wid, code_speed=2e6)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i1)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar2
    save_dir = './resource/small_sample_data_chapter2/train/radar02'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i2 in range(0, 80):
        f_c = np.random.uniform(Fs/12, Fs/10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        sequence = gen_one_bpsk(fs=Fs, carr_fre=f_c, pulse_width=pul_wid, code_speed=2e6)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i2)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar3
    save_dir = './resource/small_sample_data_chapter2/train/radar03'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i3 in range(0, 80):
        f_c = np.random.uniform(Fs/12, Fs/10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        sequence = gen_one_qpsk(fs=Fs, carr_fre=f_c, pulse_width=pul_wid, code_speed=2e6)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i3)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar4
    save_dir = './resource/small_sample_data_chapter2/train/radar04'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i4 in range(0, 80):
        f_c = np.random.uniform(Fs / 4, Fs / 3)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        sequence = gen_one_qpsk(fs=Fs, carr_fre=f_c, pulse_width=pul_wid, code_speed=2e6)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i4)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar5
    save_dir = './resource/small_sample_data_chapter2/train/radar05'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i5 in range(0, 80):
        f_c = np.random.uniform(Fs/12, Fs/10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        f_delta = np.random.uniform(Fs/6, Fs/5) - f_c  # 两个相互独立变量的运算法则
        sequence = gen_one_2fsk(fs=Fs, f_c=f_c, f_delta=f_delta, pulse_width=pul_wid, code_speed=1.2e6)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i5)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar6
    save_dir = './resource/small_sample_data_chapter2/train/radar06'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i6 in range(0, 80):
        f_c = np.random.uniform(Fs / 12, Fs / 10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        f_delta = np.random.uniform(Fs / 6, Fs / 5) - f_c  # 两个相互独立变量的运算法则
        sequence = gen_one_2fsk(fs=Fs, f_c=f_c, f_delta=f_delta, pulse_width=pul_wid, code_speed=2e6)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i6)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar7
    save_dir = './resource/small_sample_data_chapter2/train/radar07'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i7 in range(0, 80):
        f_c = np.random.uniform(Fs / 6, Fs / 5)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        sequence = gen_one_fre_sig(fs=Fs, carr_fre=f_c, pulse_width=pul_wid)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i7)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar8
    save_dir = './resource/small_sample_data_chapter2/train/radar08'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i8 in range(0, 80):
        f_c = np.random.uniform(Fs / 12, Fs / 10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        sequence = gen_one_fre_sig(fs=Fs, carr_fre=f_c, pulse_width=pul_wid)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i8)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar9
    save_dir = './resource/small_sample_data_chapter2/train/radar09'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i9 in range(0, 80):
        f_c = np.random.uniform(Fs / 10, Fs / 8)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        chirp_rate = np.random.uniform(Fs / 6, Fs / 5) / pul_wid  # 比例还与脉冲宽度有关
        sequence = gen_one_chirp_sig(fs=Fs, carr_fre=f_c, chirp_rate=chirp_rate, pulse_width=pul_wid)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i9)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar10
    save_dir = './resource/small_sample_data_chapter2/train/radar10'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i10 in range(0, 80):
        f_c = np.random.uniform(Fs / 12, Fs / 10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        chirp_rate = np.random.uniform(Fs / 12, Fs / 10) / pul_wid  # 比例还与脉冲宽度有关
        sequence = gen_one_chirp_sig(fs=Fs, carr_fre=f_c, chirp_rate=chirp_rate, pulse_width=pul_wid)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i10)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar11
    save_dir = './resource/small_sample_data_chapter2/train/radar11'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i11 in range(0, 80):
        f_c = np.random.uniform(Fs / 12, Fs / 10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        chirp_rate = np.random.uniform(Fs / 12, Fs / 10) / pul_wid  # 比例还与脉冲宽度有关
        sequence = gen_one_chirp_sig(fs=Fs, carr_fre=f_c, chirp_rate=-chirp_rate, pulse_width=pul_wid)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i11)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar12
    save_dir = './resource/small_sample_data_chapter2/train/radar12'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i12 in range(0, 80):
        f_c = np.random.uniform(Fs / 6, Fs / 5)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        v_rate = np.random.uniform(Fs / 20, Fs / 15) / pul_wid * 2  # 比例还与脉冲宽度有关
        sequence = gen_one_vee_fre_sig(fs=Fs, carr_fre=f_c, v_rate=-v_rate, pulse_width=pul_wid)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i12)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据
    return 1


def create_12_radar_test(num_test=20):
    """
    根据论文表2-1产生数据集
    """
    # radar1
    save_dir = './resource/small_sample_data_chapter2/test/radar01'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i1 in range(0, num_test):
        f_c = np.random.uniform(Fs/6, Fs/5)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        sequence = gen_one_bpsk(fs=Fs, carr_fre=f_c, pulse_width=pul_wid, code_speed=2e6)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i1)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar2
    save_dir = './resource/small_sample_data_chapter2/test/radar02'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i2 in range(0, num_test):
        f_c = np.random.uniform(Fs/12, Fs/10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        sequence = gen_one_bpsk(fs=Fs, carr_fre=f_c, pulse_width=pul_wid, code_speed=2e6)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i2)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar3
    save_dir = './resource/small_sample_data_chapter2/test/radar03'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i3 in range(0, num_test):
        f_c = np.random.uniform(Fs/12, Fs/10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        sequence = gen_one_qpsk(fs=Fs, carr_fre=f_c, pulse_width=pul_wid, code_speed=2e6)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i3)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar4
    save_dir = './resource/small_sample_data_chapter2/test/radar04'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i4 in range(0, num_test):
        f_c = np.random.uniform(Fs / 4, Fs / 3)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        sequence = gen_one_qpsk(fs=Fs, carr_fre=f_c, pulse_width=pul_wid, code_speed=2e6)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i4)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar5
    save_dir = './resource/small_sample_data_chapter2/test/radar05'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i5 in range(0, num_test):
        f_c = np.random.uniform(Fs/12, Fs/10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        f_delta = np.random.uniform(Fs/6, Fs/5) - f_c  # 两个相互独立变量的运算法则
        sequence = gen_one_2fsk(fs=Fs, f_c=f_c, f_delta=f_delta, pulse_width=pul_wid, code_speed=1.2e6)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i5)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar6
    save_dir = './resource/small_sample_data_chapter2/test/radar06'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i6 in range(0, num_test):
        f_c = np.random.uniform(Fs / 12, Fs / 10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        f_delta = np.random.uniform(Fs / 6, Fs / 5) - f_c  # 两个相互独立变量的运算法则
        sequence = gen_one_2fsk(fs=Fs, f_c=f_c, f_delta=f_delta, pulse_width=pul_wid, code_speed=2e6)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i6)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar7
    save_dir = './resource/small_sample_data_chapter2/test/radar07'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i7 in range(0, num_test):
        f_c = np.random.uniform(Fs / 6, Fs / 5)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        sequence = gen_one_fre_sig(fs=Fs, carr_fre=f_c, pulse_width=pul_wid)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i7)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar8
    save_dir = './resource/small_sample_data_chapter2/test/radar08'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i8 in range(0, num_test):
        f_c = np.random.uniform(Fs / 12, Fs / 10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        sequence = gen_one_fre_sig(fs=Fs, carr_fre=f_c, pulse_width=pul_wid)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i8)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar9
    save_dir = './resource/small_sample_data_chapter2/test/radar09'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i9 in range(0, num_test):
        f_c = np.random.uniform(Fs / 10, Fs / 8)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        chirp_rate = np.random.uniform(Fs / 6, Fs / 5) / pul_wid  # 比例还与脉冲宽度有关
        sequence = gen_one_chirp_sig(fs=Fs, carr_fre=f_c, chirp_rate=chirp_rate, pulse_width=pul_wid)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i9)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar10
    save_dir = './resource/small_sample_data_chapter2/test/radar10'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i10 in range(0, num_test):
        f_c = np.random.uniform(Fs / 12, Fs / 10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        chirp_rate = np.random.uniform(Fs / 12, Fs / 10) / pul_wid  # 比例还与脉冲宽度有关
        sequence = gen_one_chirp_sig(fs=Fs, carr_fre=f_c, chirp_rate=chirp_rate, pulse_width=pul_wid)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i10)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar11
    save_dir = './resource/small_sample_data_chapter2/test/radar11'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i11 in range(0, num_test):
        f_c = np.random.uniform(Fs / 12, Fs / 10)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        chirp_rate = np.random.uniform(Fs / 12, Fs / 10) / pul_wid  # 比例还与脉冲宽度有关
        sequence = gen_one_chirp_sig(fs=Fs, carr_fre=f_c, chirp_rate=-chirp_rate, pulse_width=pul_wid)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i11)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据

    # radar12
    save_dir = './resource/small_sample_data_chapter2/test/radar12'  # 定义保存路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建目录
        os.makedirs(save_dir)
    # 生成
    for i12 in range(0, num_test):
        f_c = np.random.uniform(Fs / 6, Fs / 5)
        pul_wid = np.random.uniform(4e-6, 6e-6)
        v_rate = np.random.uniform(Fs / 20, Fs / 15) / pul_wid * 2  # 比例还与脉冲宽度有关
        sequence = gen_one_vee_fre_sig(fs=Fs, carr_fre=f_c, v_rate=-v_rate, pulse_width=pul_wid)
        spwvd_matrix = get_spwvd(sequence, Fs)
        file_name = '{:03d}.npy'.format(i12)  # 生成文件名，格式为 000.npy - 079.npy
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, spwvd_matrix)  # 保存数据
    return 1


