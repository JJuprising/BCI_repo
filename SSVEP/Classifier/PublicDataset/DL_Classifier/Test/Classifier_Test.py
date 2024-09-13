# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/1/30 17:16
import datetime
import sys

sys.path.append('../')
import torch
import Utils.EEGDataset as EEGDataset
from Utils import Ploter
from Train import Classifier_Trainer, Trainer_Script
from etc.global_config import config


def run():
    # 1、Define parameters of eeg
    algorithm = config['algorithm'] # 本次训练的模型
    classes = config['classes'] # 本次训练所用到数据集

    print(f"{'*' * 20} Current Algorithm usage: {algorithm} Using Dataset {classes} classes {'*' * 20}")
    train_radio = 0.8
    '''Parameters for training procedure'''
    UD = config["train_param"]['UD']
    ratio = config["train_param"]['ratio']
    print(f"{'*' * 20} train_param: UD-{UD} train_radio-{train_radio} {'*' * 20}")
    # if ratio == 1 or ratio == 3:
    #     Kf = 5
    #     train_ratio = 1
    # elif ratio == 2:
    #     Kf = 2

    Kf = 1
    # 训练集占比
    if ratio == 1:
        train_radio = 0.8
    elif ratio == 2:
        train_radio = 0.5
    elif ratio == 3:
        train_radio = 0.2
    '''Parameters for ssvep data'''
    # 这两个公开数据集的采样率和注视时间是不一样的
    if classes == 12:
        ws = config["data_param_12"]["ws"] # 有效数据时间
        Ns = config["data_param_12"]['Ns'] # 被试数量
    if classes == 40:
        ws = config["data_param_40"]["ws"]
        Ns = config["data_param_40"]['Ns']

    '''Parameters for DL-based methods'''
    epochs = config[algorithm]['epochs']
    lr_jitter = config[algorithm]['lr_jitter']

    devices = "cuda" if torch.cuda.is_available() else "cpu"

    # 2、Start Training
    final_acc_list = []
    all_subsdata = None
    all_subslabel = None
    for i in range(1):
        final_test_acc_list = []
        # config["data_param_12"]["ws"] = 2.0

        for testSubject in range(1, Ns + 1): # 不同被试都要重新再一遍数据显得很冗余 [1,11)
            # **************************************** #
            '''12-class SSVEP Dataset'''
            # -----------Intra-Subject Experiments--------------
            # EEGData_Train = EEGDataset.getSSVEP12Intra(subject=testSubject, KFold=fold_num, n_splits=Kf,
            #                                            mode='train')
            # EEGData_Test = EEGDataset.getSSVEP12Intra(subject=testSubject, KFold=fold_num, n_splits=Kf,
            #                                           mode='test')
            #
            # if ratio == 3:
            #     Temp = EEGData_Train
            #     EEGData_Train = EEGData_Test
            #     EEGData_Test = Temp

            # -----------12classes Experiments--------------
            if classes == 12: # 数据集1
                # -----------12classes Intra-Subject Experiments--------------
                if UD == 0:
                    EEGData_Train = EEGDataset.getSSVEP12Intra(subject=testSubject, train_ratio=train_radio,
                                                               mode='train')
                    EEGData_Test = EEGDataset.getSSVEP12Intra(subject=testSubject, train_ratio=train_radio, mode='test')
                # -----------12classes Inter-Subject Experiments--------------
                elif UD == 1:
                    EEGData_Train = EEGDataset.getSSVEP12Inter(subject=testSubject,
                                                               mode='train')
                    EEGData_Test = EEGDataset.getSSVEP12Inter(subject=testSubject,  mode='test')
            # -----------40classes  Experiments--------------
            elif classes == 40: # 数据集2
                # -----------40classes Intra-Subject Experiments--------------
                if UD == 0:
                    EEGData_Train = EEGDataset.getSSVEP40Intra(subject=testSubject, train_ratio=train_radio,
                                                               mode='train')
                    EEGData_Test = EEGDataset.getSSVEP40Intra(subject=testSubject, train_ratio=train_radio, mode='test')
                # -----------40classes Inter-Subject Experiments--------------
                elif UD == 1:
                    EEGData_Train = EEGDataset.getSSVEP40Inter(subject=testSubject,mode='train',all_data=all_subsdata,all_labels=all_subslabel)
                    if all_subsdata is None or all_subslabel is None:
                        all_subsdata=EEGData_Train.eeg_raw_data
                        all_subslabel=EEGData_Train.label_raw_data
                    EEGData_Test = EEGDataset.getSSVEP40Inter(subject=testSubject,  mode='test',all_data=all_subsdata,all_labels=all_subslabel)


            eeg_train_dataloader, eeg_test_dataloader = Trainer_Script.data_preprocess(EEGData_Train, EEGData_Test,ws)

            # Define Network
            net, criterion, optimizer = Trainer_Script.build_model(devices,ws) # net网络结果 criterion optimizer
            # print(net)
            val_interval=1
            test_acc = Classifier_Trainer.train_on_batch(testSubject, epochs, val_interval, eeg_train_dataloader, eeg_test_dataloader, optimizer,
                                                         criterion,net, devices,ws, lr_jitter=lr_jitter)

            final_test_acc_list.append(test_acc)
            print(f"Subject {testSubject} Test Accuracy: {test_acc:.3f}")
        final_acc_list.append(final_test_acc_list)
        dataset='DatasetA' if classes==12 else 'DatasetB'
        if algorithm == 'KANformer':
            algorithm = algorithm + '/'+str(config['KANformer']['width'])
        Ploter.plot_save_Result(final_acc_list, model_name=algorithm, dataset=dataset, UD=UD, ratio=ratio,
                                win_size=str(ws), text=True)

    # print(final_acc_list)
    # 3、Plot Result


# 主函数
if __name__ == '__main__':
    run()
