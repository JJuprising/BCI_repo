import eel
import time
import datetime
# import torch
import csv
import sys
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter
from threading import Thread

# from model import vgg
board = None
board_id = None
brainflow_file_name = ''


def startSession():
    global board
    global board_id
    # board_id = 5
    board_id = 532
    params = BrainFlowInputParams()
    params.ip_port = 9521
    params.ip_address = '192.168.123.103'
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

# 打标函数
@eel.expose
def trigger(marker):
    global board
    print('marker', marker)
    board.insert_marker(marker + 1)
    return 1

# 开启
@eel.expose
def start():
    global board
    global brainflow_file_name
    dir_path = './data/'
    time_now = datetime.datetime.now()
    time_string = time_now.strftime("%Y-%m-%d_%H-%M-%S")
    brainflow_file_name = dir_path + "BrainFlow-RAW_" + time_string + '_0' + '.csv'
    # board.start_stream(45000, 'file://'+brainflow_file_name+':w')

# 停止
@eel.expose
def stop():
    global board
    data = board.get_board_data()
    dir_path = './data/'
    time_now = datetime.datetime.now()
    time_string = time_now.strftime("%Y-%m-%d_%H-%M-%S")
    brainflow_file_name = dir_path + "BrainFlow-RAW_" + time_string + '_0' + '.csv'
    DataFilter.write_file(data, brainflow_file_name, 'w')
    board.stop_stream()
    board.release_all_sessions()
    time.sleep(2)
    sys.exit(0)

def main():
    # 打开MindBridge采集流
    startSession()
    # 打开刺激范式
    eel.init('web')
    eel.start("index.html")


if __name__ == '__main__':
    main()

