"""
绘制清华数据集时域图、平铺图和复数频谱图
"""
import sys
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy import signal as SIG

def SSVEPFilter(eegData, filterType=0):
    """
    type
      0: nomal trca
      1: enhance trca
    """
    data = eegData
    fs = 250
    dataFiltered = None
    if filterType == 0:
        Wn = [7.0, 70.0]
        Wn = np.array(Wn, np.float64) / (fs / 2)
        b, a = SIG.cheby1(4, 0.1, Wn, btype="bandpass", analog=False, output='ba')
        dataFiltered = SIG.lfilter(b, a, data, axis=1)
    elif filterType == 1:
        sys.exit("Error:<filterType=1 means use Enhance trca by adding filter bank, to which Leo was lazy!!!>")
    print("eegFiltered's shape ", dataFiltered.shape)
    return dataFiltered

# 创建MNE信息对象
sub_file= r'E:\02project\BCI-Github\DL_Classifier\data\tsinghua\S32.npy'
sfreq = 250  # 采样频率
ch_names = ['Pz', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'O1', 'Oz', 'O2']
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
trails = 1

# 创建原始数据对象
data = np.load(sub_file)

# 滤波
samples = SSVEPFilter(eegData=data)

# 将数据展平到 (240, 9, 1500)
flattened_data = np.reshape(samples, (9, 1500, 240))
eeg_data = np.moveaxis(flattened_data, 2, 0)  # 变为 (240, 9, 1500)

# 通道颜色
channel_colors = plt.cm.get_cmap('tab10', len(ch_names))

for sub_data in eeg_data:
    raw = mne.io.RawArray(sub_data, info)
    raw = raw.notch_filter(freqs=(50))  # 去50hz

    # 绘制多通道时域图
    fig, ax = plt.subplots()
    for i, ch_name in enumerate(ch_names):
        ax.plot(raw.times, sub_data[i] + 100 * i, color=channel_colors(i), label=ch_name)  # 偏移以区分通道
    ax.set_title(f'Time Domain Plot {trails}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (uV)')
    ax.legend(loc='upper right')
    plt.show()

    # 绘制PSD图
    raw.plot_psd(fmin=7, fmax=20, color=channel_colors)
    plt.show()

    # 绘制复数频谱图
    complex_spectrum = np.fft.fft(sub_data, axis=1)
    freqs = np.fft.fftfreq(sub_data.shape[1], d=1/sfreq)
    fig, ax = plt.subplots()
    for i, ch_name in enumerate(ch_names):
        ax.plot(freqs, np.abs(complex_spectrum[i]), color=channel_colors(i), label=ch_name)
    ax.set_title(f'Complex Spectrum Plot {trails}')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.legend(loc='upper right')
    plt.show()

    trails += 1
