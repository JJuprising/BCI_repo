
# 独立成分分析，去伪迹，脑区图

import sys

import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
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
# ch_names = ['Channel 1']

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)
trails=1
# 创建原始数据对象
data = np.load(sub_file)
# 滤波
samples=SSVEPFilter(eegData=data)
flattened_data = np.reshape(samples, (9, 1500, 240))
eeg_data = np.moveaxis(flattened_data, 2, 0)  # 将最后一个维度移到第一个位置，变为 (240, 9, 1500)
# eeg_data=np.average(eeg_data,axis=1)
for sub_data in eeg_data:
    # sub_data=np.transpose(sub_data)
    # sub_data=sub_data.reshape(1,1500)
    raw = mne.io.RawArray(sub_data, info)
    # 创建ICA对象
    ica = ICA(n_components=6, random_state=97)
    # 拟合ICA模型
    ica.fit(raw)
    # 检查成分
    ica.plot_components()
    ica.plot_sources(raw)
    # sleep(100)
    plt.show()
