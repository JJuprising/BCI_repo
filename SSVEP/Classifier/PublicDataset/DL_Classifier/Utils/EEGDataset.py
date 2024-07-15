# Designer:Pan YuDong
# Coder:God's hand
# Time:2021/10/6 22:47
import sys

import numpy as np
from torch.utils.data import Dataset
import torch
import scipy.io
from etc.global_config import config
from scipy import signal as SIG

Ns=0 # number of subjects
classes = config['classes'] # 数据集
if classes == 12:
    Ns = config["data_param_12"]['Ns']
elif classes == 40:
    Ns = config["data_param_40"]['Ns']
'''
tsinghua benckmark datasets 40
    _FREQS = [
        8, 9, 10, 11, 12, 13, 14, 15, 
        8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2, 
        8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
        8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
        8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8
    ]
'''

class getSSVEP12Inter(Dataset):
    def __init__(self, subject=1, mode="train"):
        self.Nh = 180
        self.Nc = 8
        self.Nt = 1024
        self.Nf = 12
        self.Fs = 256
        self.eeg_raw_data = self.read_EEGData()
        self.label_raw_data = self.read_EEGLabel()
        if mode == 'train':
            self.eeg_data = torch.cat(
                (self.eeg_raw_data[0:(subject - 1) * self.Nh], self.eeg_raw_data[subject * self.Nh:]), dim=0)
            self.label_data = torch.cat(
                (self.label_raw_data[0:(subject - 1) * self.Nh:, :], self.label_raw_data[subject * self.Nh:, :]), dim=0)

        if mode == 'test':
            self.eeg_data = self.eeg_raw_data[(subject - 1) * self.Nh:subject * self.Nh]
            self.label_data = self.label_raw_data[(subject - 1) * self.Nh:subject * self.Nh]

        print(f'eeg_data for subject {subject}:', self.eeg_data.shape)
        print(f'label_data for subject {subject}:', self.label_data.shape)

    def __getitem__(self, index):
        return self.eeg_data[index], self.label_data[index]

    def __len__(self):
        return len(self.label_data)
    def SSVEPFilter(self, eegData,filterType=0):
        """
        type
          0:
          1:
        """
        data = eegData
        # print("eegData's shape ", self._eegData.shape)
        fs = self.Fs
        dataFiltered = None
        if filterType == 0:
            Wn = [8.0, 64.0]
            Wn = np.array(Wn, np.float64) / (fs / 2)
            b, a = SIG.cheby1(4, 0.1, Wn, btype="bandpass", analog=False, output='ba')
            dataFiltered = SIG.lfilter(b, a, data, axis=1)
            #            dataFiltered = SIG.filtfilt(b, a, data, axis = 1)
            del b, a, Wn
        elif filterType == 1:
            sys.exit("Error:<filterType=1 means use Enhance trca by adding filter bank, to which Leo was lazy!!!>")
        # print("eegFiltered's shape ", dataFiltered.shape)
        # del data, dataFiltered
        return dataFiltered
    # get the single subject data
    def get_DataSub(self, index):
        # load file into dict
        subjectfile = scipy.io.loadmat(f'../data/Dial/DataSub_{index}.mat')
        # extract numpy from dict
        samples = subjectfile['Data'] # (8, 1024, 180)
        samples=self.SSVEPFilter(samples,filterType=0)
        # (num_trial, sample_point, num_trial) => (num_trial, num_channels, sample_point)
        eeg_data = samples.swapaxes(1, 2)
        eeg_data = torch.from_numpy(eeg_data.swapaxes(0, 1))
        eeg_data = eeg_data.reshape(-1, 1, self.Nc, self.Nt)
        # print(eeg_data.shape)
        return eeg_data

    # 所有数据拼起来
    def read_EEGData(self):
        eeg_data = self.get_DataSub(1)
        for i in range(1, Ns):
            single_subject_eeg_data = self.get_DataSub(i + 1)
            eeg_data = torch.cat((eeg_data, single_subject_eeg_data), dim=0)
        return eeg_data

    # get the single label data
    def get_DataLabel(self, index):
        # load file into dict
        labelfile = scipy.io.loadmat(f'../data/Dial/LabSub_{index}.mat')
        # extract numpy from dict
        labels = labelfile['Label']
        label_data = torch.from_numpy(labels)
        # print(label_data.shape)
        return label_data - 1

    def read_EEGLabel(self):
        label_data = self.get_DataLabel(1)
        for i in range(1, Ns):
            single_subject_label_data = self.get_DataLabel(i)
            label_data = torch.cat((label_data, single_subject_label_data), dim=0)
        return label_data


class getSSVEP12Intra(Dataset):
    def __init__(self, subject=1, train_ratio=0.8, KFold=None, n_splits=5, mode="train"):
        super(getSSVEP12Intra, self).__init__()
        self.Nh = 180  # number of trials
        self.Nc = 8  # number of channels
        self.Nt = 1024  # number of time points
        self.Nf = 12  # number of target frequency
        self.Fs = 256  # Sample Frequency
        self.subject = subject  # current subject
        self.eeg_data = self.get_DataSub()
        self.label_data = self.get_DataLabel()
        self.num_trial = self.Nh // self.Nf  # number of trials of each frequency
        self.train_idx = []
        self.test_idx = []
        if KFold is not None:
            fold_trial = self.num_trial // n_splits  # number of trials in each fold
            self.valid_trial_idx = [i for i in range(KFold * fold_trial, (KFold + 1) * fold_trial)]

        for i in range(0, self.Nh, self.Nh // self.Nf):
            for j in range(self.Nh // self.Nf):
                if n_splits == 2 and j == self.num_trial - 1:
                    continue  # if K = 2, discard the last trial of each category
                if KFold is not None:  # K-Fold Cross Validation
                    if j not in self.valid_trial_idx:
                        self.train_idx.append(i + j)
                    else:
                        self.test_idx.append(i + j)
                else:  # Split Ratio Validation
                    if j < int(self.num_trial * train_ratio):
                        self.train_idx.append(i + j)
                    else:
                        self.test_idx.append(i + j)

        self.eeg_data_train = self.eeg_data[self.train_idx]
        self.label_data_train = self.label_data[self.train_idx]
        self.eeg_data_test = self.eeg_data[self.test_idx]
        self.label_data_test = self.label_data[self.test_idx]

        if mode == 'train':
            self.eeg_data = self.eeg_data_train
            self.label_data = self.label_data_train
        elif mode == 'test':
            self.eeg_data = self.eeg_data_test
            self.label_data = self.label_data_test

        print(f'eeg_data for subject {subject}:', self.eeg_data.shape)
        print(f'label_data for subject {subject}:', self.label_data.shape)

    def __getitem__(self, index):
        return self.eeg_data[index], self.label_data[index]

    def __len__(self):
        return len(self.label_data)

    # get the single subject data
    def get_DataSub(self):
        subjectfile = scipy.io.loadmat(f'../data/Dial/DataSub_{self.subject}.mat')
        samples = subjectfile['Data']  # (8, 1024, 180)
        eeg_data = samples.swapaxes(1, 2)  # (8, 1024, 180) -> (8, 180, 1024)
        eeg_data = torch.from_numpy(eeg_data.swapaxes(0, 1))  # (8, 180, 1024) -> (180, 8, 1024)
        eeg_data = eeg_data.reshape(-1, 1, self.Nc, self.Nt)  # (180, 1, 8, 1024)
        # print(eeg_data.shape)
        return eeg_data

    # get the single label data
    def get_DataLabel(self):
        labelfile = scipy.io.loadmat(f'../data/Dial/LabSub_{self.subject}.mat')
        labels = labelfile['Label']
        # print(labels)
        label_data = torch.from_numpy(labels)
        # print(label_data.shape)  # torch.Size([180, 1])
        return label_data - 1


# 清华跨被试数据处理
class getSSVEP40Inter(Dataset):
    def __init__(self, subject=1,  mode="train",all_data=None,all_labels=None):
        super(getSSVEP40Inter, self).__init__()
        self.Nh = 240  # number of trials 6 blocks x 40 trials
        self.Nc = 9  # number of channels
        self.Nt = 1500  # number of time points
        self.Nf = 40  # number of target frequency
        self.Fs = 250  # Sample Frequency
        if subject==1: # 第一个要加载数据
            self.eeg_raw_data = self.read_EEGData()
            self.label_raw_data = self.read_EEGLabel()
        else:
            self.eeg_raw_data=all_data
            self.label_raw_data=all_labels
        if mode == 'train': # 对比每一个被试进行重载
            self.eeg_data = torch.cat(
                (self.eeg_raw_data[0:(subject - 1) * self.Nh], self.eeg_raw_data[subject * self.Nh:]), dim=0)
            self.label_data = torch.cat(
                (self.label_raw_data[0:(subject - 1) * self.Nh:, :], self.label_raw_data[subject * self.Nh:, :]), dim=0)
        if mode == 'test':
            self.eeg_data = self.eeg_raw_data[(subject - 1) * self.Nh:subject * self.Nh]
            self.label_data = self.label_raw_data[(subject - 1) * self.Nh:subject * self.Nh]

        print(f'eeg_data for subject {subject}:', self.eeg_data.shape)
        print(f'label_data for subject {subject}:', self.label_data.shape)


    def __getitem__(self, index):
        return self.eeg_data[index], self.label_data[index]

    def __len__(self):
        return len(self.label_data)

    def SSVEPFilter(self, eegData,filterType=0):
        """
        type
          0:
          1:
        """
        data = eegData
        # print("eegData's shape ", self._eegData.shape)
        fs = self.Fs
        dataFiltered = None
        if filterType == 0:
            Wn = [8.0, 64.0]
            Wn = np.array(Wn, np.float64) / (fs / 2)
            b, a = SIG.cheby1(4, 0.1, Wn, btype="bandpass", analog=False, output='ba')
            dataFiltered = SIG.lfilter(b, a, data, axis=1)
            #            dataFiltered = SIG.filtfilt(b, a, data, axis = 1)
            del b, a, Wn
        elif filterType == 1:
            sys.exit("Error:<filterType=1 means use Enhance trca by adding filter bank, to which Leo was lazy!!!>")
        # print("eegFiltered's shape ", dataFiltered.shape)
        # del data, dataFiltered
        return dataFiltered

    def get_DataSub(self,index):
        # subjectfile = scipy.io.loadmat(f'../data/tsinghua/S{index}.mat')
        # samples = subjectfile['data']  # (64, 1500, 40, 6)
        # # 做一个通道选择
        # # O1, Oz, O2, PO3, POZ, PO4, PZ, PO5 and PO6 对应 61, 62, 63, 55, 56, 57, 48, 54, 58
        # chnls = [48, 54, 55, 56, 57, 58, 61, 62, 63]
        # samples = samples[chnls, :, :, :]
        samples=np.load(r'../data/tsinghua/S' + str(index) + r'.npy') # (9,1500,40,6)
        # 滤波
        samples = self.SSVEPFilter(eegData=samples)
        # print(f'subject {self.subject} data shape: {samples.shape}')
        # 处理格式
        # 0x6,1x6...
        # 将数据展平到 (240, 9, 1500)
        flattened_data = np.reshape(samples, (9, 1500, 240))
        eeg_data = np.moveaxis(flattened_data, 2, 0)  # 将最后一个维度移到第一个位置，变为 (240, 9, 1500)
        np.clip(eeg_data, -12, 12, out=eeg_data)
        # 处理格式
        # data = samples.transpose((3, 2, 0, 1))  # (6, 40, 64, 1500)
        # data_stack = [data[i:i + 1, :, :, :].squeeze(0) for i in range(data.shape[0])]  # (1, 40, 64, 1500)
        # eeg_data = np.vstack(data_stack)  # (240,64,1500)
        eeg_data = eeg_data.reshape(-1, 1, self.Nc, self.Nt)  # (240, 1, 64, 1500) # 这个第二维度什么作用？
        eeg_data = torch.from_numpy(eeg_data)
        # print(eeg_data.shape)  # (trails, 1, channels, times)
        return eeg_data

    def read_EEGData(self):
        eeg_data = self.get_DataSub(1)
        for i in range(1, Ns):
            single_subject_eeg_data = self.get_DataSub(i + 1)
            eeg_data = torch.cat((eeg_data, single_subject_eeg_data), dim=0)
        return eeg_data

    # get the single label data
    def get_DataLabel(self):
        # label_array = [[i] for i in np.tile(np.arange(self.Nf), 6)]  # 从 0 到 39 个trails循环 6 个block，每个元素是单个数组
        # label_data = torch.tensor(label_array)
        # 生成从0到39的数组
        # base_array = np.arange(40)  # [0, 1, 2, ..., 39]
        # # 重复6次以生成6个区块
        # label_data=[[i] for i in np.repeat(base_array, 6)]

        label_data=np.load(r'../data/tsinghua/S1_label.npy') # 这里不影响，所有label顺序都是一样的
        label_data = torch.tensor(label_data)  # 每个元素重复6次

        # print(label_data.shape)
        return label_data  # 不需要和12分类一样-1，这里已经是从0开始了

    def read_EEGLabel(self):
        label_data = self.get_DataLabel()
        for i in range(0, Ns-1):
            single_subject_label_data = self.get_DataLabel()
            label_data = torch.cat((label_data, single_subject_label_data), dim=0)
        return label_data


# 基于清华数据集 实验由6个区块组成。每个区块包含40次试验对应40目标，持续5s
# 时间点要不要直接切掉休息的0.5s
class getSSVEP40Intra(Dataset):
    def __init__(self, subject=1, train_ratio=0.8, KFold=None, n_splits=5, mode="train"):
        super(getSSVEP40Intra, self).__init__()
        self.Nh = 240  # number of trials 6 blocks x 40 trials
        self.Nc = 9  # number of channels
        self.Nt = 1500  # number of time points
        self.Nf = 40  # number of target frequency
        self.Fs = 250  # Sample Frequency
        self.subject = subject  # current subject
        self.eeg_data = self.get_DataSub()
        self.label_data = self.get_DataLabel()
        self.num_trial = self.Nh // self.Nf  # number of trials of each frequency
        self.train_idx = []
        self.test_idx = []
        if KFold is not None:
            fold_trial = self.num_trial // n_splits  # number of trials in each fold
            self.valid_trial_idx = [i for i in range(KFold * fold_trial, (KFold + 1) * fold_trial)]

        for i in range(0, self.Nh, self.Nh // self.Nf):
            for j in range(self.Nh // self.Nf):
                if n_splits == 2 and j == self.num_trial - 1:
                    continue  # if K = 2, discard the last trial of each category
                if KFold is not None:  # K-Fold Cross Validation
                    if j not in self.valid_trial_idx:
                        self.train_idx.append(i + j)
                    else:
                        self.test_idx.append(i + j)
                else:  # Split Ratio Validation
                    if j < int(self.num_trial * train_ratio):
                        self.train_idx.append(i + j)
                    else:
                        self.test_idx.append(i + j)

        self.eeg_data_train = self.eeg_data[self.train_idx]
        self.label_data_train = self.label_data[self.train_idx]
        self.eeg_data_test = self.eeg_data[self.test_idx]
        self.label_data_test = self.label_data[self.test_idx]
        print(self.label_data)
        if mode == 'train':
            self.eeg_data = self.eeg_data_train
            self.label_data = self.label_data_train
        elif mode == 'test':
            self.eeg_data = self.eeg_data_test
            self.label_data = self.label_data_test

        print(f'eeg_data for subject {subject}:', self.eeg_data.shape)
        print(f'label_data for subject {subject}:', self.label_data.shape)


    def __getitem__(self, index):
        return self.eeg_data[index], self.label_data[index]

    def __len__(self):
        return len(self.label_data)

    def SSVEPFilter(self, eegData,filterType=0):
        """
        type
          0: nomal trca
          1: enhance trca
        """
        data = eegData
        # print("eegData's shape ", self._eegData.shape)
        fs = self.Fs
        dataFiltered = None
        if filterType == 0:
            Wn = [6.0, 90.0]
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
    # get the single subject data
    def get_DataSub(self):
        # 加载mat
        # subjectfile = scipy.io.loadmat(f'../data/tsinghua/S{self.subject}.mat')
        # samples = subjectfile['data']  # (64, 1500, 40, 6)
        # # 做一个通道选择
        # # O1, Oz, O2, PO3, POZ, PO4, PZ, PO5 and PO6 对应 61, 62, 63, 55, 56, 57, 48, 54, 58
        # chnls = [48, 54, 55, 56, 57, 58, 61, 62, 63]
        # samples = samples[chnls, :, :, :]
        # 直接记载处理好的npy
        samples = np.load(r'../data/tsinghua/S' + str(self.subject) + r'.npy')  # (9,1500,40,6)
        # 滤波
        # samples=self.SSVEPFilter(eegData=samples)
        # print(f'subject {self.subject} data shape: {samples.shape}')
        # 处理格式

        # 0x6,1x6...
        # 将数据展平到 (240, 9, 1500)
        flattened_data = np.reshape(samples, (9, 1500, 240))
        eeg_data = np.moveaxis(flattened_data, 2, 0)  # 将最后一个维度移到第一个位置，变为 (240, 9, 1500)

        # 0-39 x6
        # data = samples.transpose((3, 2, 0, 1))  # (6, 40, 64, 1500)
        # data_stack = [data[i:i + 1, :, :, :].squeeze(0) for i in range(data.shape[0])]  # (1, 40, 64, 1500)
        # eeg_data = np.vstack(data_stack)  # (240,64,1500)


        eeg_data = eeg_data.reshape(-1, 1, self.Nc, self.Nt)  # (240, 1, 64, 1500) # 这个第二维度什么作用？
        eeg_data = torch.from_numpy(eeg_data)
        print(eeg_data.shape)  # (trails, 1, channels, times)
        return eeg_data

    # get the single label data
    def get_DataLabel(self):
        # 0x6,1x6...
        label_data = np.load(r'../data/tsinghua/S1_label.npy')  # 这里不影响，所有label顺序都是一样的
        label_data = torch.tensor(label_data)  # 每个元素重复6次


        print(label_data.shape)
        return label_data  # 不需要和12分类一样-1，这里已经是从0开始了
