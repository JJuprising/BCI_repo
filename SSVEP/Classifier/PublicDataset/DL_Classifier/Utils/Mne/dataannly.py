'''
绘制多通道时域图和PSD图
'''

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
    # print("eegData's shape ", self._eegData.shape)
    fs = 250
    dataFiltered = None
    if filterType == 0:
        Wn = [7.0, 70.0]
        Wn = np.array(Wn, np.float64) / (fs / 2)
        b, a = SIG.cheby1(4, 0.1, Wn, btype="bandpass", analog=False, output='ba')
        dataFiltered = SIG.lfilter(b, a, data, axis=1)
        #            dataFiltered = SIG.filtfilt(b, a, data, axis = 1)
        del b, a, Wn
    elif filterType == 1:
        sys.exit("Error:<filterType=1 means use Enhance trca by adding filter bank, to which Leo was lazy!!!>")
    print("eegFiltered's shape ", dataFiltered.shape)
    # del data, dataFiltered
    return dataFiltered
# 创建MNE信息对象
sub_file= r'E:\02project\BCI-Github\DL_Classifier\data\tsinghua\S11.npy'
sfreq = 250  # 采样频率
ch_names = ['Pz', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'O1', 'Oz', 'O2']
# ch_names = ['Pz']
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
trails=1
# 创建原始数据对象
data = np.load(sub_file)
# 滤波
samples=SSVEPFilter(eegData=data)
# print(f'subject {self.subject} data shape: {samples.shape}')
# 处理格式

# 0x6,1x6...
# 将数据展平到 (240, 9, 1500)
flattened_data = np.reshape(samples, (9, 1500, 240))
eeg_data = np.moveaxis(flattened_data, 2, 0)  # 将最后一个维度移到第一个位置，变为 (240, 9, 1500)
# eeg_data=np.average(eeg_data,axis=1) # 配套
for sub_data in eeg_data:

    # sub_data=np.transpose(sub_data)# 配套
    # sub_data=sub_data.reshape(1,1500)# 配套
    raw = mne.io.RawArray(sub_data, info)
    raw = raw.notch_filter(freqs=(50)) # 去50hz
    # 绘制多通道时域图
    raw.plot(n_channels=len(ch_names), scalings='auto', title=f'Time Domain Plot{trails}')
    # 绘制PSD图
    raw.plot_psd(fmin=7,fmax=20)
    plt.show()
    trails+=1
