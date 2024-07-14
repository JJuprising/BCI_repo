import numpy as np 
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations, DetrendOperations
from scipy import signal
from matplotlib import pyplot as plt

'''
这个代码与之前的annly.py不同在于做了感兴趣频率限制
由于信号分散等原因，效果不是很好
把功率谱限制在感兴趣的频率段中，以增强效果
'''


# data = np.loadtxt('./data/fkl-9-ssvep-8-10-13-15-17-21-18-11-23/BrainFlow-RAW_2024-05-06_14-56-05_0.csv').T
data = np.loadtxt(r'E:\02project\BCI-Github\BCI-UAV\BCIcode\quickssvep-master\pythondata\3x14-3x16-3x18-3x20\cyj\BrainFlow-RAW_2024-04-09_21-55-14_0.csv').T
print(data.shape)

labels = data[-2]
print(labels[labels != 0])
# Blink Frequency Array 刺激块数组
Blink_freq = [14,16,18,20]
# 通道
chnls=[2,3]
# eeg_data = data[[24,25,26, 28, 29,30]]
# eeg_data = data[[24,28, 29,30]]
eeg_data = data[chnls]


# egg_data = data[1]
# eeg_data = np.loadtxt('12hz_2.txt').tolist()
# eeg_data = np.array(eeg_data)
# print(eeg_data.shape)
# labels = [1]
for channel in range(len(eeg_data)):
    DataFilter.detrend(eeg_data[channel], DetrendOperations.NO_DETREND.value)
    DataFilter.perform_bandpass(eeg_data[channel], 1000, 7, 24, 8,
                                        FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_bandstop(eeg_data[channel], 1000, 48, 52, 8,
                                        FilterTypes.BUTTERWORTH.value, 0)
# eeg_data[1] = eeg_data[1] -eeg_data[0]

# 感兴趣频率段进行叠加
def zsnr( patpowavedata, foi_idx, number):

    pxx_all = []
    for index in foi_idx:
        pxx_c = []
        for i in range(index, index +2):
            pxx_c.append(patpowavedata[i])
        b = np.sum(pxx_c)
        pxx_all.append(b)
    return pxx_all, foi_idx

for i in range(1):
    for index in range(len(labels)):
        label = labels[index]
        if label !=0 :
            subData = eeg_data[:,index+0000: index+4000] # 数据长度
            pxx_all = []
            freq_a = []
            for channel in range(len(subData)):
                if channel == 0:
                    continue
                [freq, pxx] = signal.welch(subData[channel], 1000, nperseg=5120, average="median", window='hann')
                freq_a = freq
                pxx_all.append(pxx)
            freq = freq[0: 120]

            # stim_freq = [13, 15, 17, 18, 11, 21]
            stim_freq = Blink_freq
            indexs = [freq.tolist().index(item) for item in stim_freq]
            pxx = np.mean(np.array(pxx_all), axis=0)[0:120]
            patsnr, foi_idx = zsnr(np.array(pxx), indexs, 1)
            pxx_all = []
            for index in range(len(freq)):
                pxx_all.append(0)
            for item in range(len(indexs)):
                pxx_all[indexs[item]] = patsnr[item]
            print(np.max(pxx_all), freq[np.where(pxx_all == np.max(pxx_all))])
            plt.plot(freq, pxx_all)
            plt.show()