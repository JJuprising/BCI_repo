import os
import numpy as np
import pandas as pd
from brainflow import DataFilter, DetrendOperations, FilterTypes

label_index = -2  # 8ch是-2 32ch是-1
chanels = [[2, 3]]
dur_gaze = 4  # data length for target identification [s]
delay = 0.13
sampling_rate = 1000
down_sample = 250
# 指定文件夹路径
folder_path = r'../3x14-3x16-3x18-3x20/cyj'


def read_csv_files_in_folder(folder_path):
    """
    读取指定文件夹下的所有CSV文件，并将它们合并为一个NumPy数组
    """
    data_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = np.loadtxt(file_path).T  # 转置后 (通道，采样点)
            # print(df.values.shape)
            eeg_data = df[chanels]  # 通道数据
            labels = df[label_index]  # 打标列
            eeg_data = processBydatafilter(eeg_data)  # 预处理
            use_data = getData(eeg_data, labels)  # 数据切片
            print("use_data", use_data.shape)
            data_list.append(use_data)  # 将DataFrame转换为NumPy数组并添加到列表中
    if data_list:
        return np.concatenate(data_list, axis=0)  # 合并所有数组
    else:
        print("没有找到任何CSV文件！")
        return None


# 根据label提取有效数据进行数据切片
def getData(eeg_data, labels):
    # 把第一个放进来
    use_data = []
    count = 0
    for index in range(len(labels)):
        label = labels[index]
        if label != 0:
            if count == 0:
                # 把第一个塞进来
                font_data = eeg_data[:, index - 6000: index - 3000]
                use_data.append(font_data)
            subData = eeg_data[:, index + 1000: index + dur_gaze * sampling_rate]
            use_data.append(subData)
            count+=1
    return np.array(use_data)


# 传统预处理方法
def processBydatafilter(eeg_data):
    for channel in range(len(eeg_data)):
        DataFilter.detrend(eeg_data[channel], DetrendOperations.NO_DETREND.value)
        DataFilter.perform_bandpass(eeg_data[channel], 1000, 10, 21, 2,
                                    FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(eeg_data[channel], 1000, 48, 52, 2,
                                    FilterTypes.BUTTERWORTH.value, 0)
    return eeg_data


def main():
    """
    主函数
    """
    data = read_csv_files_in_folder(folder_path)
    if data is not None:
        # print("读取到的数据：", data)
        print(data.shape)
        # 将数组保存为 .npy 文件
        np.save('../3x14-3x16-3x18-3x20/cyj/cyj_pySSVEP_3000.npy', data)


if __name__ == '__main__':
    main()
