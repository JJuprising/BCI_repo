# from os import times
'''
FDR矫正

'''
import sys
import numpy as np
from matplotlib import pyplot as plt
from mne.stats import fdr_correction
from scipy.stats import ttest_rel, ttest_ind
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
s1='11'
s2='32'
sub1_file=r'E:\02project\BCI-Github\DL_Classifier\data\tsinghua\S'+s1+'.npy'
sub2_file=r'E:\02project\BCI-Github\DL_Classifier\data\tsinghua\S'+s2+'.npy'
data1=np.load(sub1_file)
data2=np.load(sub2_file)
data1=SSVEPFilter(eegData=data1)
data2=SSVEPFilter(eegData=data2)
# 将数据展平到 (240, 9, 1500)
flattened_data1 = np.reshape(data1, (9, 1500, 240))
eeg_data1 = np.moveaxis(flattened_data1, 2, 0)  # 将最后一个维度移到第一个位置，变为 (240, 9, 1500)
flattened_data2 = np.reshape(data2, (9, 1500, 240))
eeg_data2 = np.moveaxis(flattened_data2, 2, 0)  # 将最后一个维度移到第一个位置，变为 (240, 9, 1500)
times=np.arange(0,1500)
for sub1, sub2 in zip(eeg_data1, eeg_data2):
    # 独立样本t检验
    t_vals, p_vals = ttest_ind(sub1, sub2, axis=0)
    # FDR矫正
    rejects, p_fdr_corrected = fdr_correction(p_vals, alpha=0.05)
    # 可视化经过矫正的统计检验结果
    plt.plot(times, np.average(sub1, axis=0), label=s1)
    plt.plot(times, np.average(sub2, axis=0), label=s2)
    # for i, reject in enumerate(rejects):
    #     if reject.any() == True:
    #         plt.axvline(x=times[i], color='grey', alpha=0.2)
    plt.legend()
    plt.show()


