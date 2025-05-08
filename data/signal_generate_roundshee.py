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


def spwvd_fft_optimized(signal, fs, g, h):
    length_signal = len(signal)
    window_length_g = len(g)
    window_length_h = len(h)
    max_tau_g = window_length_g // 2
    max_tau_h = window_length_h // 2
    M = length_signal
    K = window_length_g // 2 + 1  # Frequency bins based on window g
    spwvd_result = np.zeros((K, M), dtype=np.complex128)

    # Precompute the center index for the windows
    g_center = max_tau_g
    h_center = max_tau_h

    for m in nb.prange(M):
        # Determine valid tau range considering both g window and signal boundaries
        tau_max = min(max_tau_g, m, length_signal - 1 - m)
        tau = np.arange(-tau_max, tau_max + 1)
        terms = np.zeros(window_length_g, dtype=np.complex128)

        # Compute signal product terms with g window
        for i, ta in enumerate(tau):
            idx1 = m + ta
            idx2 = m - ta
            if idx1 >= 0 and idx1 < length_signal and idx2 >= 0 and idx2 < length_signal:
                g_idx = ta + g_center
                terms[g_idx] = g[g_idx] * signal[idx1] * np.conj(signal[idx2])

        # Perform FFT on the terms to get frequency components
        fft_result = np.fft.fft(terms)
        fft_positive = fft_result[:K]

        # Apply h window in time domain
        # Determine valid theta range considering h window
        theta_min = max(-max_tau_h, -m)
        theta_max = min(max_tau_h, length_signal - 1 - m)
        for theta in range(theta_min, theta_max + 1):
            h_idx = theta + h_center
            if 0 <= h_idx < window_length_h:
                t_shifted = m + theta
                if 0 <= t_shifted < M:
                    spwvd_result[:, t_shifted] += h[h_idx] * fft_positive

    return spwvd_result


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
    g = np.array(windows.gaussian(window_length, std=window_length/6))
    h = np.array(windows.gaussian(window_length, std=window_length/6))
    # out_spwvd = spwvd_fft_optimized(signal, fs, g, h)
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


def gen_one_bpsk(fs, carr_fre, pulse_width, code_speed, random_phase=1, code_rand=True):
    """
    根据成电论文，生成指定参数下，模拟接收到的中频处理后的信号，处理后载波为2倍输入值。但根据论文提供的时频图以及测试参数，
    这里的bpsk为直接接收结果，不考虑2倍处理。 码信息随机生成
    :param code_rand:
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
    if code_rand:  # 默认码随机
        codes = np.random.randint(0, 2, size=one_pulse_code_nums)  # 生成随机码序列 BPSK是randint(0,2)不含2
        if np.all(codes == 0):
            codes[-1] = 1  # 将最后一个元素改为1
        elif np.all(codes == 1):
            codes[-1] = 0  # 将最后一个元素改为0
    else:
        base_pattern = [0, 1]
        codes = np.resize(base_pattern, one_pulse_code_nums).astype(int)
    codes = np.repeat(codes, code_fs_num)  # 生成码序列对应脉冲相位
    codes = codes[:output_len]  # 截取与脉冲长度对应
    out_phase = 2*np.pi*carr_fre*output_time + phi_0 + np.pi*codes  # 相位计算
    out_bpsk = np.cos(out_phase)
    return out_bpsk


def gen_one_qpsk(fs, carr_fre, pulse_width, code_speed, random_phase=1, code_rand=True):
    """
    如上QPSK
    :param code_rand:
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
    if code_rand:
        codes = np.random.randint(0, 4, size=one_pulse_code_nums)  # 生成随机码序列
    else:
        base_pattern = [0, 1, 2, 3]
        codes = np.resize(base_pattern, one_pulse_code_nums).astype(int)
    codes = np.repeat(codes, code_fs_num)  # 生成码序列对应脉冲相位
    codes = codes[:output_len]  # 截取与脉冲长度对应
    out_phase = 2*np.pi*carr_fre*output_time + phi_0 + np.pi*(codes/2+0.25)  # 相位计算
    out_qpsk = np.cos(out_phase)
    return out_qpsk


def gen_one_2fsk(fs, f_c, f_delta, pulse_width, code_speed, random_phase=1, code_rand=True):
    """
    2FSK生成 频率表达式：f(t)=f_c+f_delta*code(tau)
    :param code_rand:
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
    if code_rand:
        codes = np.random.randint(0, 2, size=one_pulse_code_nums)  # 生成随机码序列 BPSK是randint(0,2)不含2
        if np.all(codes == 0):
            codes[-1] = 1  # 将最后一个元素改为1
        elif np.all(codes == 1):
            codes[-1] = 0  # 将最后一个元素改为0
    else:
        base_pattern = [0, 1]
        codes = np.resize(base_pattern, one_pulse_code_nums).astype(int)
    codes = np.repeat(codes, code_fs_num)  # 生成码序列对应脉冲相位
    codes = codes[:output_len]  # 截取与脉冲长度对应
    out_phase = 2*np.pi*(f_c+f_delta*codes)*output_time+phi_0
    out_2fsk = np.cos(out_phase)
    return out_2fsk


def gen_frank_code(fs, carr_freq, pulse_width, M=4, random_phase=True):
    """
    生成Frank编码信号
    :param fs: 采样率 (Hz)
    :param carr_freq: 载频 (Hz)
    :param pulse_width: 脉宽 (秒)
    :param M: Frank码阶数 (默认4)
    :param random_phase: 是否添加随机初始相位
    :return: Frank编码信号 (numpy数组)
    """
    # 基础参数计算
    total_samples = int(fs * pulse_width)
    time_axis = np.arange(total_samples) / fs

    # 初始化相位
    phi_0 = np.random.uniform(0, 2 * np.pi) if random_phase else 0

    # 生成Frank相位矩阵
    frank_matrix = np.zeros((M, M))
    for i in range(M):
        for k in range(M):
            frank_matrix[i, k] = (2 * np.pi * (i * k)) / M

    # 展平为相位序列
    phase_sequence = frank_matrix.flatten()

    # 计算码元参数
    num_chips = M ** 2  # 总码元数
    chip_duration = pulse_width / num_chips
    samples_per_chip = int(fs * chip_duration)

    # 生成重复的相位序列
    repeated_phase = np.repeat(phase_sequence, samples_per_chip)

    # 对齐信号长度
    final_phase = np.resize(repeated_phase, total_samples)

    # 合成信号
    signal = np.cos(2 * np.pi * carr_freq * time_axis + phi_0 + final_phase)

    return signal


def gen_p1_code(fs, carr_freq, pulse_width, M=4, random_phase=True):
    """
    生成P1编码信号（多普勒容限优化的多相码）
    :param fs: 采样率 (Hz)
    :param carr_freq: 载频 (Hz)
    :param pulse_width: 脉宽 (秒)
    :param M: P1码阶数 (默认4)
    :param random_phase: 是否添加随机初始相位
    :return: P1编码信号 (numpy数组)
    """
    # 基础参数计算
    total_samples = int(fs * pulse_width)  # 总采样点数
    time_axis = np.arange(total_samples) / fs  # 时间轴

    # 初始化相位
    phi_0 = np.random.uniform(0, 2 * np.pi) if random_phase else 0

    # 生成P1相位序列
    phase_sequence = np.zeros(M ** 2)
    for i in range(1, M + 1):  # i从1到M
        for m in range(1, M + 1):  # m从1到M
            idx = (i - 1) * M + (m - 1)  # 序列索引
            phase_sequence[idx] = -np.pi * (m - 1) * (2 * i - 1 - M) / M ** 2

    # 计算码元参数
    num_chips = M ** 2  # 总码元数
    chip_duration = pulse_width / num_chips  # 单个码元时长
    samples_per_chip = int(fs * chip_duration)  # 每个码元的采样点数

    # 重复相位序列并截断对齐
    repeated_phase = np.repeat(phase_sequence, samples_per_chip)
    final_phase = np.resize(repeated_phase, total_samples)

    # 合成信号
    signal = np.cos(2 * np.pi * carr_freq * time_axis + phi_0 + final_phase)

    return signal


def gen_p2_code(fs, carr_freq, pulse_width, M=4, random_phase=True):
    """
    生成P2编码信号（多普勒容限优化的多相码）
    :param fs: 采样率 (Hz)
    :param carr_freq: 载频 (Hz)
    :param pulse_width: 脉宽 (秒)
    :param M: P2码阶数 (默认4)
    :param random_phase: 是否添加随机初始相位
    :return: P2编码信号 (numpy数组)
    """
    # 基础参数计算
    total_samples = int(fs * pulse_width)  # 总采样点数
    time_axis = np.arange(total_samples) / fs  # 时间轴

    # 初始化相位
    phi_0 = np.random.uniform(0, 2 * np.pi) if random_phase else 0

    # 生成P2相位序列（核心公式）
    phase_sequence = np.zeros(M ** 2)
    for i in range(1, M + 1):  # i从1到M
        for m in range(1, M + 1):  # m从1到M
            idx = (i - 1) * M + (m - 1)  # 序列索引
            phase_sequence[idx] = (np.pi / M ** 2) * (i - 1) * (2 * m - 1 - M)

    # 计算码元参数
    num_chips = M ** 2  # 总码元数
    chip_duration = pulse_width / num_chips  # 单个码元时长
    samples_per_chip = int(fs * chip_duration)  # 每个码元的采样点数

    # 重复相位序列并截断对齐
    repeated_phase = np.repeat(phase_sequence, samples_per_chip)
    final_phase = np.resize(repeated_phase, total_samples)

    # 合成信号
    signal = np.cos(2 * np.pi * carr_freq * time_axis + phi_0 + final_phase)

    return signal


def gen_p3_code(fs, carr_freq, pulse_width, M=4, random_phase=True):
    """
    生成P3编码信号（基于LFM相位量化的多相码）
    :param fs: 采样率 (Hz)
    :param carr_freq: 载频 (Hz)
    :param pulse_width: 脉宽 (秒)
    :param M: 码阶（相位量化阶数，通常为2的幂次）
    :param random_phase: 是否添加随机初始相位
    :return: P3编码信号 (numpy数组)
    """
    # 基础参数计算
    total_samples = int(fs * pulse_width)  # 总采样点数
    time_axis = np.arange(total_samples) / fs  # 时间轴

    # 初始化相位
    phi_0 = np.random.uniform(0, 2 * np.pi) if random_phase else 0

    # 生成P3相位序列（核心公式）
    num_chips = M ** 2  # 总码元数
    phase_sequence = np.zeros(num_chips)
    for k in range(num_chips):
        phase_sequence[k] = (np.pi / num_chips) * k ** 2  # 二次相位量化

    # 计算码元参数
    chip_duration = pulse_width / num_chips  # 单个码元时长
    samples_per_chip = int(fs * chip_duration)  # 每个码元的采样点数

    # 重复相位序列并截断对齐
    repeated_phase = np.repeat(phase_sequence, samples_per_chip)
    final_phase = np.resize(repeated_phase, total_samples)

    # 合成信号
    signal = np.cos(2 * np.pi * carr_freq * time_axis + phi_0 + final_phase)

    return signal


def gen_p4_code(fs, carr_freq, pulse_width, M=4, random_phase=True):
    """
    生成P4编码信号（多普勒容限优化的二次相位编码）
    :param fs: 采样率 (Hz)
    :param carr_freq: 载频 (Hz)
    :param pulse_width: 脉宽 (秒)
    :param M: 码阶（控制相位量化精细度，建议为偶数）
    :param random_phase: 是否添加随机初始相位
    :return: P4编码信号 (numpy数组)
    """
    # 基础参数计算
    total_samples = int(fs * pulse_width)  # 总采样点数
    time_axis = np.arange(total_samples) / fs  # 时间轴

    # 初始化相位
    phi_0 = np.random.uniform(0, 2 * np.pi) if random_phase else 0

    # 生成P4相位序列（核心公式）
    num_chips = M ** 2
    phase_sequence = np.zeros(num_chips)
    for k in range(num_chips):
        phase_sequence[k] = (np.pi / M ** 2) * k * (k + 1)  # 关键公式

    # 将相位限制在[0, 2π)范围内
    phase_sequence = np.mod(phase_sequence, 2 * np.pi)

    # 计算码元参数
    chip_duration = pulse_width / num_chips  # 单个码元时长
    samples_per_chip = int(fs * chip_duration)  # 每个码元的采样点数

    # 重复相位序列并截断对齐
    repeated_phase = np.repeat(phase_sequence, samples_per_chip)
    final_phase = np.resize(repeated_phase, total_samples)

    # 合成信号
    signal = np.cos(2 * np.pi * carr_freq * time_axis + phi_0 + final_phase)

    return signal



