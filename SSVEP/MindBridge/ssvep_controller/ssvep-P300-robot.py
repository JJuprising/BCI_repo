import sys
import threading
from random import random

import joblib
import pywt
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPixmap, QPainter, QPen, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout
import socket
import json
from threading import Thread
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QPen
import random
import time
# from naoqi import ALProxy
import cv2
import numpy as np

from BCIcode.SSVEP_cnn import SSVEPNET

global h
h = 0
global start_time
start_time = 0
global t1, t2, t3, t4, t5, t6
t1 = t2 = t3 = t4 = t5 = t6 = 0
from TRCA import TRCA

client_socket = None
window = None
characters = ['./image/抓取', './image/上.png', './image/释放', './image/左.png', './image/右.png', './image/下.png']

# 闪烁频率
frequencies = [9.25 + i * 0.5 for i in range(len(characters))]  # 据字符数量计算得出的，从6开始，每个字符的闪烁频率增加0.5
global svm_model
svm_model = joblib.load('./svm_model.pkl')

import argparse
import time

import mne
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WaveletTypes
from scipy import signal as SIG
from scipy.signal import resample
# from ../SSVEP_cnn/SSVEPNET/SCUJJ.py import SCU
from SCUJJ import SCU
sample_rate = 1000  # 采样率
downsample_rate = 250  # 降采样Hz
chn_nums = 6  # 6通道
channels = [0, 1, 2, 3, 4, 5]  # 通道

# def SsvepCheck2():

board = None


def main():
    global board
    BoardShim.enable_dev_board_logger()

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=True)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                        required=False, default=BoardIds.NO_BOARD)
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file
    params.master_board = args.master_board

    board = BoardShim(args.board_id, params)
    board.prepare_session()
    board.start_stream()
    record_data = True

    while True:
        # python ssvep-P300-robot.py --ip-address 192.168.137.81 --board-id 5 --ip-port 9527
        time.sleep(2)  # Wait for 2 second
        print("start")
        if record_data:
            data = board.get_current_board_data(3000)  # Take data for 3 seconds
        else:
            data = None  # If not recording, set data to None
            print("no data")

        record_data = not record_data  # Toggle the flag for the next iteration

        if data is not None:
            sub_data = prepare(data)  # Preprocess the data
            print("sub_data", sub_data.shape)
            # out = SsvepCheck(sub_data)
            ssvep_out=SsvepCheck(sub_data)
            print("ssvep_out", ssvep_out)
            judge(ssvep_out, sub_data)

        # print(data.shape)
    board.stop_stream()
    board.release_session()
    datafilter = DataFilter()
    # data = board.get_board_data()
    # datafilter.write_file(data, "./realtime.csv", "w")


def SsvepCheck(data):
    # 小波去噪用不了，不知道为什么
    # for count, channel in enumerate(channels):
    #      DataFilter.perform_wavelet_denoising(sub_data[channel], wavelet=4,decomposition_level=5,threshold=20)

    # 小波去噪 -cyj

    eeg_data = data
    for channel_item in range(len(eeg_data)):
        item = eeg_data[channel_item]
        coeffs = pywt.wavedec(item, 'db4', level=4)  # 使用 'db4' 小波基对当前通道数据进行4层小波分解
        threshold = np.sqrt(2 * np.log(len(data)))  # 根据原始数据长度计算阈值
        thresholded_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]  # 阈值处理 将小于阈值的系数置为0，保留大于阈值的系数。
        reconstructed_signal = pywt.waverec(thresholded_coeffs, 'db4')  # 使用 'db4' 小波基将阈值处理后的系数重构为信号。
        eeg_data[channel_item] = reconstructed_signal[0: len(eeg_data[channel_item])]  # 将去噪后的信号存储回原来的 EEG 通道数据中

    data = eeg_data.T

    for i in range(len(data)):
        # 预处理
        DataFilter.detrend(data[i], DetrendOperations.CONSTANT.value)  # 去趋势
        # 带宽滤波 6-30Hz
        DataFilter.perform_bandpass(data[i], downsample_rate, 6.0, 30.0, 2,
                                    FilterTypes.BUTTERWORTH.value, 0)
        # 带相滤波 去除 48.0 Hz 到 52.0 Hz 和 58.0 Hz 到 62.0 Hz 的频率成分
        DataFilter.perform_bandstop(data[i], downsample_rate, 48.0, 52.0, 2,
                                    FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(data[i], downsample_rate, 58.0, 62.0, 2,
                                    FilterTypes.BUTTERWORTH.value, 0)

    data = data.T

    # 降采样
    target_time_points = int(3000)  # 重采样
    sub_data = resample(data, target_time_points)
    sub_data = sub_data.T
    out=use_SCU(sub_data)
    # 不知道你TRCA要的是几维，顺序是什么 (6,1500)
    # out = use_TRCA(testdata=sub_data)
    return out


def P300Check(P300sub_data):
    features = []
    features.append(P300sub_data)
    features = np.array(features)

    # 分类
    print(features.shape)

    global svm_model
    # 预测
    prediction = svm_model.predict(features)
    if prediction == 1:
        print("有P300")
        return True
    else:
        print("没有检测到P300")
        return False


def judge(out, data):
    global t1, t2, t3, t4, t5, t6
    if out == 1:
        t = t1 * 1000
    elif out == 2:
        t = t2 * 1000
    elif out == 3:
        t = t3 * 1000
    elif out == 4:
        t = t4 * 1000
    elif out == 5:
        t = t5 * 1000
    elif out == 6:
        t = t6 * 1000

    origin_data = np.mean(data.T, axis=0)
    print("origin_data", origin_data.shape)
    data_mean = np.mean(origin_data)
    data_std = np.std(origin_data)
    origin_data = (origin_data - data_mean) / data_std
    origin_data = np.array(origin_data.tolist())
    DataFilter.detrend(origin_data, DetrendOperations.CONSTANT.value)
    DataFilter.calc_stddev(origin_data)

    # calc_stddev()函数用于计算给定数组的标准差（Standard Deviation），标准差是衡量数据离散程度的一种指标，它越小表示数据越集中，越大表示数据越分散。
    # 这个滤波有待商榷，因为报警告
    # 报错原因是filter_length的长度默认是6.6倍的sfreq，原本是6601但是数据只有2000，超出了，解决方法把sfreq调小吧
    def_data = mne.filter.filter_data(origin_data, sfreq=100, l_freq=0.5, h_freq=1.5, verbose=False)
    origin_data = origin_data - def_data
    P300sub_data = origin_data[t + 50:t + 800]

    print("P300sub_Data", P300sub_data.shape)

    if P300Check(P300sub_data) == True:
        print("输出结果", out)
    else:
        print("有误或在空闲中")


# 预处理
def prepare(origin_data=None):
    # 带通
    data = origin_data[channels]
    for count, channel in enumerate(channels):
        DataFilter.perform_bandpass(data[channel], sample_rate, 1.0, 50.0, 4,
                                    FilterTypes.BESSEL.value, 0)

    # target_time_points = int(3000 * downsample_rate / sample_rate)  # 所有数据*1/4
    # data = resample(data, target_time_points)

    # 滤波
    # origin_data = mne.filter.filter_data(data, sfreq=1000, l_freq=0.1, h_freq=90, verbose=False)  # 带宽滤波到0.1-90hz
    dataFiltered = data.T

    return dataFiltered

def use_SCU(testdata):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs of training')
    parser.add_argument('--lr', type=float, default=0.001, help='adam: learning rate')  # 0.00001
    parser.add_argument('--dropout_level', type=float, default=0.55, help='dropout level')
    parser.add_argument('--w_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--seed_n', type=int, default=74, help='seed number')
    opt = parser.parse_args()

   # 目前四分类
    num_classes = 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 创建一个新的模型实例
    cnn = SCU(opt, num_classes=num_classes).to(device)

    # 加载已保存的参数
    cnn.load_state_dict(torch.load('../SSVEP_cnn/SSVEPNET/models/SSVEPnetSCU_93.75.pth'))

    # 设置为评估模式
    cnn.eval()

    # inputs = preprocess_realtime_data(inputs) 输入为(2,3000)
    inputs = torch.from_numpy(testdata).float().to(device)

    # 添加批量维度并进行模型推理
    with torch.no_grad():
        inputs = inputs.unsqueeze(0)  # 添加批量维度
        # 输入格式(1,2,3000)
        outputs = cnn(inputs)
        print("outputs:", outputs) # 这里就是每个类别的参数
        _, predicted = torch.max(outputs, 1)
        predicted_label = predicted.item()
        print("ssvep Predicted Label:", predicted_label)

    return outputs




def use_TRCA(testdata=None):
    trca = TRCA()
    print("test", testdata.shape)
    # 这里记得要改
    out = trca.realClassifier(testdata, 4) + 1  # 分类，预测结果
    print("trca answer is:" + str(out))
    return out

def trigger(marker):
    global board
    print('marker', marker)
    board.insert_marker(marker + 1)
    return 1

class CharacterWidget(QWidget):
    def __init__(self, character, frequency):
        super().__init__()
        self.flag = True
        self.character = character
        self.frequency = frequency
        self.label = QLabel(self)
        pixmap = QPixmap(character)  # 替换为图片文件路径
        scaled_pixmap = pixmap.scaled(100, 100)  # 替换为固定大小
        self.label.setPixmap(scaled_pixmap)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: white; color: black;")
        self.label.setFont(QFont("Arial", 50, QFont.Bold))
        self.label.setFixedSize(100, 100)

        self.timer = QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.toggle_visibility)
        self.timer.start(1000 / frequency)
        # 定时对数据进行刷新
        self.isVisible = True

    def toggle_visibility(self):
        if self.flag == True:
            self.isVisible = not self.isVisible
            self.label.setVisible(self.isVisible)

    def toggle_Flash(self):
        self.label.setVisible(True)
        if self.flag == True:
            self.flag = False
        else:
            self.flag = True


class MainWindow(QWidget):
    my_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.character_widgets = []
        self.mainLayout = QVBoxLayout(self)
        self.centerWidget = QWidget(self)
        self.blink_duration = 2  # 闪烁
        self.pause_duration = 2  # 暂停
        self.current_index = 0
        self.stopFlag = False
        self.setFixedSize(1200, 730)

        # 打标值
        self.marker = 1

        # self.my_signal.connect(self.recv)
        self.init_ui()
        self.start_blinking()

    def init_ui(self):
        self.setStyleSheet("background-color: black;")
        # 设置容器的布局

        self.p300Widget = P300Widget()
        self.p300Widget.setParent(self)

        # 增加第一行按钮的布局
        h1Layout = QHBoxLayout()
        h1Layout.addSpacing(100)
        h1Layout.setSpacing(130)
        character = characters[0]
        frequency = frequencies[0]
        self.widget1 = CharacterWidget(character, frequency)
        self.character_widgets.append(self.widget1)

        character = characters[1]
        frequency = frequencies[1]
        self.widget2 = CharacterWidget(character, frequency)
        self.character_widgets.append(self.widget2)

        character = characters[2]
        frequency = frequencies[2]
        self.widget3 = CharacterWidget(character, frequency)
        self.character_widgets.append(self.widget3)
        h1Layout.addWidget(self.widget1)
        h1Layout.addWidget(self.widget2)
        h1Layout.addWidget(self.widget3)

        # 增加第二行按钮的布局
        h2Layout = QHBoxLayout()
        h2Layout.addSpacing(100)
        h2Layout.setSpacing(520)
        character = characters[3]
        frequency = frequencies[3]
        self.widget4 = CharacterWidget(character, frequency)
        self.character_widgets.append(self.widget4)

        # self.cameraWidget = VideoWidget()
        character = characters[4]
        frequency = frequencies[4]
        self.widget5 = CharacterWidget(character, frequency)
        self.character_widgets.append(self.widget5)

        h2Layout.addWidget(self.widget4)
        # h2Layout.addWidget(self.cameraWidget)
        h2Layout.addWidget(self.widget5)

        # 增加第三行按钮的布局
        h3Layout = QHBoxLayout()
        h3Layout.addSpacing(500)

        character = characters[5]
        frequency = frequencies[5]
        self.widget6 = CharacterWidget(character, frequency)
        self.character_widgets.append(self.widget6)
        h3Layout.addWidget(self.widget6)

        self.mainLayout.addSpacing(70)
        self.mainLayout.setSpacing(100)
        # 将三行水平布局加入主垂直布局
        self.mainLayout.addLayout(h1Layout)
        self.mainLayout.addLayout(h2Layout)
        self.mainLayout.addLayout(h3Layout)
        self.setLayout(self.mainLayout)

        self.setWindowTitle('SSVEP-Robot')

        self.show()

    def start_blinking(self):
        if self.stopFlag == True:
            return

        self.blink_characters()

        QTimer.singleShot(2 * 1000, self.start_blinking)  # 这里加起来才是闪烁/暂停时间

    def stop_blinking(self):
        self.stopFlag == True

    def blink_characters(self):
        global client_socket
        self.current_index += 1

        for widget in self.character_widgets:
            widget.toggle_Flash()

        self.p300Widget.setTrue()  # 更改P300的闪烁和暂停状态

        # 打标
        # if client_socket != None:
        #     client_socket.sendall(json.dumps({"action": "trigger", "data": self.current_index}).encode('utf-8'))
        # print(self.marker)
        # trigger(self.marker)
        # self.marker+=1

    def closeEvent(self, event):
        event.accept()


class P300Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(150, 150)
        self.setVisible(False)
        self.flag = True
        self.timer = QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.update_widget_position)
        # (490,60)为1，上,(490,550)为2下，(90,306)为3，左,(885,306)为4，右，(90,60)为5,抓取,(895,60)为6，释放
        self.position = [(90, 60), (490, 60), (895, 60), (90, 306), (885, 306), (490, 550)]
        self.i = 0
        self.show_duration = 1200  # Duration in milliseconds to show the widget
        self.flash_duration = 100  # Duration in milliseconds for each flash
        self.flash_interval = 100  # Interval in milliseconds between flashes
        self.timer.start(200)

    # 绘制P300的形状
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.white, 20, Qt.SolidLine))
        width = self.width()
        height = self.height()
        painter.drawRect(0, 0, width - 2, height - 2)

    # 更新P300的位置
    def update_widget_position(self):
        if self.flag == False:
            self.setVisible(False)
        else:
            if self.i * (self.flash_interval + self.flash_duration) < self.show_duration:
                self.setVisible(True)
                if self.i % 6 == 0:
                    global start_time
                    start_time = time.perf_counter()
                    global h
                    h += 1
                    print("第{}轮".format(h))
                    self.remaining_elements = self.position.copy()

                if self.remaining_elements:
                    end_time = time.perf_counter()
                    chosen_element = random.choice(self.remaining_elements)
                    self.remaining_elements.remove(chosen_element)
                    self.move(*chosen_element)
                    self.i += 1
                    print(chosen_element)
                    time_interval = end_time - start_time
                    if chosen_element == (490, 60):
                        t1 = time_interval
                        print("1", t1)
                    elif chosen_element == (490, 550):
                        t2 = time_interval
                        print("2", t2)
                    elif chosen_element == (90, 306):
                        t3 = time_interval
                        print("3", t3)
                    elif chosen_element == (885, 306):
                        t4 = time_interval
                        print("4", t4)
                    elif chosen_element == (90, 60):
                        t5 = time_interval
                        print("5", t5)
                    elif chosen_element == (895, 60):
                        t6 = time_interval
                        print("6", t6)

            else:
                QTimer.singleShot(800, lambda: self.reset_i())
                self.setVisible(False)

    def reset_i(self):
        self.i = 0

    def setTrue(self):
        if self.flag == True:

            self.flag = False
            self.setVisible(False)
        else:
            self.flag = True


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 脑电范式
    threadSocket = Thread(target=main)
    threadSocket.start()
    # 开启范式
    window = MainWindow()
    sys.exit(app.exec_())
