# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/1/31 10:53
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from Model import EEGNet, CCNN, SSVEPNet, FBtCNN, ConvCA, SSVEPformer, DDGCNN, CNNBIGRU, CAIFormer, FBSSVEPformer, \
    FBTFformer3,TFFFormer

from Model import EEGNet, CCNN, SSVEPNet, FBtCNN, ConvCA, SSVEPformer, DDGCNN, CNNBIGRU, CNNAttentionGRU, \
    CNNAttentionMLP, CACAM, CACAMNew, PSDCNN, CAIFormerNew, TFformer, iTransformer, KANformer, TFFBformer, SSVEPformer2, SSVEPformer3,TFformerAtt,TFformer3

from Utils import Constraint, LossFunction, Script
from etc.global_config import config
import torch.nn.functional as F


def normalize_data(train_data, test_data):
    # Flatten the data
    train_data_flattened = train_data.reshape(train_data.shape[0], -1)
    test_data_flattened = test_data.reshape(test_data.shape[0], -1)

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit on training data
    scaler.fit(train_data_flattened)

    # Transform both train and test data
    train_data_normalized_flattened = scaler.transform(train_data_flattened)
    test_data_normalized_flattened = scaler.transform(test_data_flattened)

    # Reshape back to original shape
    train_data_normalized = train_data_normalized_flattened.reshape(train_data.shape)
    test_data_normalized = test_data_normalized_flattened.reshape(test_data.shape)

    return train_data_normalized, test_data_normalized
def data_preprocess(EEGData_Train, EEGData_Test,ws):
    '''
    Parameters
    ----------
    EEGData_Train: EEG Training Dataset (Including Data and Labels)
    EEGData_Test: EEG Testing Dataset (Including Data and Labels)

    Returns: Preprocessed EEG DataLoader
    -------
    '''
    algorithm = config['algorithm']
    classes=config['classes']

    last_time = 0  # 各数据中的延迟时间
    if classes == 12:
        # ws = config["data_param_12"]["ws"]
        Fs = config["data_param_12"]["Fs"]
        Nf = config["data_param_12"]["Nf"]
        last_time = 0.135
    elif classes == 40:
        # ws = config["data_param_40"]["ws"]
        Fs = config["data_param_40"]["Fs"]
        Nf = config["data_param_40"]["Nf"]
        last_time = 0.64

    bz = config[algorithm]["bz"]

    '''Loading Training Data'''
    EEGData_Train, EEGLabel_Train = EEGData_Train[:]
    EEGData_Train = EEGData_Train[:, :, :, int(Fs * last_time):int(Fs * ws) +int(Fs * last_time)]

    '''Loading Testing Data'''
    EEGData_Test, EEGLabel_Test = EEGData_Test[:]
    EEGData_Test = EEGData_Test[:, :, :, int(Fs * last_time):int(Fs * ws) +int(Fs * last_time)]

    EEGData_Train, EEGData_Test = normalize_data(EEGData_Train, EEGData_Test)

    EEGData_Train = torch.from_numpy(EEGData_Train)
    EEGData_Test = torch.from_numpy(EEGData_Test)

    '''Training Data'''
    if algorithm == "ConvCA":
        EEGData_Train = torch.swapaxes(EEGData_Train, axis0=2, axis1=3) # (Nh, 1, Nt, Nc)
        EEGTemp_Train = Script.get_Template_Signal(EEGData_Train, Nf)  # (Nf × 1 × Nt × Nc)
        EEGTemp_Train = torch.swapaxes(EEGTemp_Train, axis0=0, axis1=1)  # (1 × Nf × Nt × Nc)
        EEGTemp_Train = EEGTemp_Train.repeat((EEGData_Train.shape[0], 1, 1, 1))  # (Nh × Nf × Nt × Nc)
        EEGTemp_Train = torch.swapaxes(EEGTemp_Train, axis0=1, axis1=3)  # (Nh × Nc × Nt × Nf)

        # print("EEGData_Train.shape", EEGData_Train.shape)
        # print("EEGTemp_Train.shape", EEGTemp_Train.shape)
        # print("EEGLabel_Train.shape", EEGLabel_Train.shape)
        EEGData_Train = torch.utils.data.TensorDataset(EEGData_Train, EEGTemp_Train, EEGLabel_Train)

    else:
        if algorithm == "CCNN":
            EEGData_Train = CCNN.complex_spectrum_features(EEGData_Train.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Train = torch.from_numpy(EEGData_Train)

        elif algorithm == "SSVEPformer":
            EEGData_Train = SSVEPformer.complex_spectrum_features(EEGData_Train.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Train = torch.from_numpy(EEGData_Train)
            EEGData_Train = EEGData_Train.squeeze(1)

        elif algorithm == "SSVEPformer2":
            EEGData_Train = SSVEPformer2.complex_spectrum_features(EEGData_Train.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Train = torch.from_numpy(EEGData_Train)
            # EEGData_Train = EEGData_Train.squeeze(1)

        elif algorithm == "SSVEPformer3":
            EEGData_Train = SSVEPformer3.complex_spectrum_features(EEGData_Train.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Train = torch.from_numpy(EEGData_Train)
            # EEGData_Train = EEGData_Train.squeeze(1)

        elif algorithm == "KANformer":
            EEGData_Train = KANformer.complex_spectrum_features(EEGData_Train.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Train = torch.from_numpy(EEGData_Train)
            EEGData_Train = EEGData_Train.squeeze(1)
        # elif algorithm == "VIT":
        #     EEGData_Train = EEGData_Train.squeeze(1)
        #     EEGData_Train = VIT.extract_freq_domain_features(EEGData_Train.numpy())
        elif algorithm == "FBSSVEPformer":
            subband_data = []
            for i in range(3):
                lowcut = i * 8 + 2 # 根据数据集 1 的特性设置截止频率
                highcut = 80
                filtered_data = FBSSVEPformer.butter_bandpass_filter(EEGData_Train, lowcut, highcut, 256)
                filtered_data = FBSSVEPformer.complex_spectrum_features(filtered_data, FFT_PARAMS=[Fs, ws]) # 数据维度：(36, 1, 8, 560)
                filtered_data = torch.from_numpy(filtered_data)
                subband_data.append(filtered_data)
            # 需要转换维度为(36, 8, 3, 256)
            EEGData_Train = torch.stack(subband_data, dim=3)
            EEGData_Train = EEGData_Train.squeeze(1)
        elif algorithm == "FBTFformer3":
            subband_data = []
            for i in range(3):
                if classes == 12:
                    lowcut = i * 8 + 2
                    highcut = 80
                elif classes == 40:
                    lowcut = i * 9 + 2
                    highcut = 80
                filtered_data = FBTFformer3.butter_bandpass_filter(EEGData_Train, lowcut, highcut, Fs)
                filtered_data = torch.from_numpy(filtered_data)
                subband_data.append(filtered_data)
            EEGData_Train = torch.stack(subband_data, dim=3)
            EEGData_Train = EEGData_Train.squeeze(1)

        elif algorithm == "DDGCNN":
            EEGData_Train = torch.swapaxes(EEGData_Train, axis0=1, axis1=3)  # (Nh, 1, Nc, Nt) => (Nh, Nt, Nc, 1)
        elif algorithm=="CACAMNew":
            EEGData_Train = CACAMNew.ssvep_to_wavelet_spectrogram(EEGData_Train.squeeze(1).numpy(),Fs)
            EEGData_Train = torch.from_numpy(EEGData_Train)

        elif algorithm=="PSDCNN":

            EEGData_Train = PSDCNN.calculate_sliding_psd(EEGData_Train.numpy(),Fs)
            EEGData_Train = torch.from_numpy(EEGData_Train).unsqueeze(1)

        print("EEGData_Train.shape", EEGData_Train.shape)
        print("EEGLabel_Train.shape", EEGLabel_Train.shape)
        EEGData_Train = torch.utils.data.TensorDataset(EEGData_Train, EEGLabel_Train)




    '''Testing Data'''
    if algorithm == "ConvCA":
        EEGData_Test = torch.swapaxes(EEGData_Test, axis0=2, axis1=3)  # (Nh, 1, Nt, Nc)
        EEGTemp_Test = Script.get_Template_Signal(EEGData_Test, Nf)  # (Nf × 1 × Nt × Nc)
        EEGTemp_Test = torch.swapaxes(EEGTemp_Test, axis0=0, axis1=1)  # (1 × Nf × Nt × Nc)
        EEGTemp_Test = EEGTemp_Test.repeat((EEGData_Test.shape[0], 1, 1, 1))  # (Nh × Nf × Nt × Nc)
        EEGTemp_Test = torch.swapaxes(EEGTemp_Test, axis0=1, axis1=3)  # (Nh × Nc × Nt × Nf)

        print("EEGData_Test.shape", EEGData_Test.shape)
        print("EEGTemp_Test.shape", EEGTemp_Test.shape)
        print("EEGLabel_Test.shape", EEGLabel_Test.shape)
        EEGData_Test = torch.utils.data.TensorDataset(EEGData_Test, EEGTemp_Test, EEGLabel_Test)

    else:
        if algorithm == "CCNN":
            EEGData_Test = CCNN.complex_spectrum_features(EEGData_Test.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Test = torch.from_numpy(EEGData_Test)

        elif algorithm == "SSVEPformer":
            EEGData_Test = SSVEPformer.complex_spectrum_features(EEGData_Test.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Test = torch.from_numpy(EEGData_Test)
            EEGData_Test = EEGData_Test.squeeze(1)

        elif algorithm == "SSVEPformer2":
            EEGData_Test = SSVEPformer2.complex_spectrum_features(EEGData_Test.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Test = torch.from_numpy(EEGData_Test)
            # EEGData_Test = EEGData_Test.squeeze(1)
        elif algorithm == "SSVEPformer3":
            EEGData_Test = SSVEPformer3.complex_spectrum_features(EEGData_Test.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Test = torch.from_numpy(EEGData_Test)
            # EEGData_Test = EEGData_Test.squeeze(1)

        elif algorithm == "KANformer":
            EEGData_Test = KANformer.complex_spectrum_features(EEGData_Test.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Test = torch.from_numpy(EEGData_Test)
            EEGData_Test = EEGData_Test.squeeze(1)

        elif algorithm == "FBSSVEPformer":
            subband_data = []
            for i in range(3):
                lowcut = i * 8 + 2# 根据数据集 1 的特性设置截止频率
                highcut = 80
                filtered_data = FBSSVEPformer.butter_bandpass_filter(EEGData_Test, lowcut, highcut, 256)
                filtered_data = FBSSVEPformer.complex_spectrum_features(filtered_data, FFT_PARAMS=[Fs, ws])
                filtered_data = torch.from_numpy(filtered_data)
                subband_data.append(filtered_data)
            EEGData_Test = torch.stack(subband_data, dim=3)
            EEGData_Test = EEGData_Test.squeeze(1)
        elif algorithm == "FBTFformer3":
            subband_data = []
            for i in range(3):
                if classes == 12:
                    lowcut = i * 8 + 2
                    highcut = 80
                elif classes == 40:
                    lowcut = i * 9 + 2
                    highcut = 80
                filtered_data = FBTFformer3.butter_bandpass_filter(EEGData_Test, lowcut, highcut, Fs)
                filtered_data = torch.from_numpy(filtered_data)
                subband_data.append(filtered_data)
            EEGData_Test = torch.stack(subband_data, dim=3)
            EEGData_Test = EEGData_Test.squeeze(1)
        elif algorithm == "DDGCNN":
            EEGData_Test = torch.swapaxes(EEGData_Test, axis0=1, axis1=3)  # (Nh, 1, Nc, Nt) => (Nh, Nt, Nc, 1)
        elif algorithm == "CACAMNew":
            EEGData_Test = CACAMNew.ssvep_to_wavelet_spectrogram(EEGData_Test.squeeze(1).numpy(),Fs)
            EEGData_Test = torch.from_numpy(EEGData_Test)
        elif algorithm == "PSDCNN":
            EEGData_Test = PSDCNN.calculate_sliding_psd(EEGData_Test.numpy(), Fs)
            EEGData_Test = torch.from_numpy(EEGData_Test).unsqueeze(1)
        # elif algorithm == "VIT":
        #     EEGData_Test=   EEGData_Test.squeeze(1)
        #     EEGData_Test = VIT.extract_freq_domain_features(EEGData_Test.numpy())


        print("EEGData_Test.shape", EEGData_Test.shape)
        print("EEGLabel_Test.shape", EEGLabel_Test.shape)
        EEGData_Test = torch.utils.data.TensorDataset(EEGData_Test, EEGLabel_Test)

    # Create DataLoader for the Dataset
    eeg_train_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Train, batch_size=bz, shuffle=True,
                                                   drop_last=True)
    eeg_test_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Test, batch_size=bz, shuffle=False,
                                                   drop_last=True)

    return eeg_train_dataloader, eeg_test_dataloader

def build_model(devices,ws):
    '''
    Parameters
    ----------
    device: the device to save DL models
    Returns: the building model
    -------
    '''
    algorithm = config['algorithm']
    classes = config['classes']
    if classes == 12:
        Nc = config["data_param_12"]['Nc']
        Nf = config["data_param_12"]['Nf']
        Fs = config["data_param_12"]['Fs']
        # ws = config["data_param_12"]['ws']
    elif classes == 40:
        Nc = config["data_param_40"]['Nc']
        Nf = config["data_param_40"]['Nf']
        Fs = config["data_param_40"]['Fs']
        # ws = config["data_param_40"]['ws']

    lr = config[algorithm]['lr']
    wd = config[algorithm]['wd']
    Nt = int(Fs * ws)

    class LabelSmoothingLoss(nn.Module):
        "Implement label smoothing."

        def __init__(self, class_num=classes, smoothing=0.01):
            super(LabelSmoothingLoss, self).__init__()
            self.confidence = 1.0 - smoothing
            self.smoothing = smoothing
            self.class_num = class_num

        def forward(self, x, target):
            assert x.size(1) == self.class_num
            if self.smoothing == None:
                return nn.CrossEntropyLoss()(x, target)

            true_dist = x.data.clone()
            true_dist.fill_(self.smoothing / (self.class_num - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

            logprobs = F.log_softmax(x, dim=-1)
            mean_loss = -torch.sum(true_dist * logprobs) / x.size(-2)
            return mean_loss
    if algorithm == "EEGNet":
        net = EEGNet.EEGNet(Nc, Nt, Nf)

    elif algorithm == "CCNN":
        net = CCNN.CNN(Nc, 220, Nf)

    elif algorithm =="CACAMNew":
        net = CACAMNew.ESNet(Nc, 6)
        net = Constraint.Spectral_Normalization(net)

    elif algorithm == "FBtCNN":
        net = FBtCNN.tCNN(Nc, Nt, Nf, Fs)

    elif algorithm == "ConvCA":
        net = ConvCA.convca(Nc, Nt, Nf)

    elif algorithm == "PSDCNN":
        net = PSDCNN.ESNet(Nc,Nt,Nf)

    elif algorithm == "SSVEPformer":
        net = SSVEPformer.SSVEPformer(depth=2, attention_kernal_length=31, chs_num=Nc, class_num=Nf,
                                      dropout=0.5)
        net.apply(Constraint.initialize_weights)

    elif algorithm == "SSVEPformer2":
        net = SSVEPformer2.SSVEPformer2(depth=2, attention_kernal_length=31, chs_num=Nc, class_num=Nf,
                                      dropout=0.5)
        net = Constraint.Spectral_Normalization(net)

    elif algorithm == "SSVEPformer3":
        net = SSVEPformer3.SSVEPformer3(depth=2, attention_kernal_length=31, chs_num=Nc, class_num=Nf,
                                      dropout=0.5)
        net.apply(Constraint.initialize_weights)

    elif algorithm == "KANformer":
        net = KANformer.KANformer(depth=2, attention_kernal_length=31, chs_num=Nc, class_num=Nf,
                                      dropout=0.5, width=config['KANformer']['width'])
        net = Constraint.Spectral_Normalization(net)

    elif algorithm == "FBSSVEPformer":
        net = FBSSVEPformer.FB_SSVEPformer(depth=2, attention_kernal_length=31, chs_num=Nc, class_num=Nf,
                                      dropout=0.5, num_subbands=3)
        net.apply(Constraint.initialize_weights)

    elif algorithm == "TFformer":
        net = TFformer.TFformer(depth=2, heads=8, chs_num=Nc, class_num=Nf, tt_dropout=0.3, ff_dropout=0.5,
                                T=Nt)
        net = Constraint.Spectral_Normalization(net)
    elif algorithm == "TFformer3":
        net = TFformer3.TFformer3(depth=2, heads=8, chs_num=Nc, class_num=Nf, tt_dropout=0.3, ff_dropout=0.5,
                                T=Nt)
        net = Constraint.Spectral_Normalization(net)
    elif algorithm=="FBTFformer3":
        net = FBTFformer3.FBTFformer3(depth=2, heads=8, chs_num=Nc, class_num=Nf, tt_dropout=0.3, ff_dropout=0.5,
                                  T=Nt, num_subbands=3)
        net = Constraint.Spectral_Normalization(net)
    elif algorithm == "TFFBformer":
        net = TFFBformer.TFFBformer(depth=2, heads=8, chs_num=Nc, class_num=Nf, tt_dropout=0.3, ff_dropout=0.5,
                                T=Nt)
        net = Constraint.Spectral_Normalization(net)
    elif algorithm == "TFformerAtt":
        net = TFformerAtt.TFformerAtt(depth=2, heads=8, chs_num=Nc, class_num=Nf, tt_dropout=0.3, ff_dropout=0.5,
                                T=Nt)
        net = Constraint.Spectral_Normalization(net)
    elif algorithm=="TFFFormer":
        net=TFFFormer.TFFFormer(device=devices,heads=4,N=1,chs_num=Nc,class_num=Nf,T=Nt)
        # net = Constraint.Spectral_Normalization(net)
    elif algorithm == "iTransformer":
        net = iTransformer.iTransformer(depth=2, heads=8, chs_num=Nc, class_num=Nf, tt_dropout=0.3, ff_dropout=0.5,
                                T=Nt)
        net = Constraint.Spectral_Normalization(net)

    elif algorithm == "CAIFormer":
        net = CAIFormer.ESNet(Nc, Nt, Nf)
        net = Constraint.Spectral_Normalization(net)

    elif algorithm == "CAIFormer":
        net = CAIFormer.ESNet(Nc, Nt, Nf)
        net = Constraint.Spectral_Normalization(net)

    elif algorithm == "CAIFormerNew":

        net = CAIFormerNew.iTransformerFFT(depth=2, heads=8, chs_num=Nc, class_num=Nf, tt_dropout=0.3, ff_dropout=0.5,
                                T=Nt)
        net = Constraint.Spectral_Normalization(net)

    elif algorithm == "SSVEPNet":
        net = SSVEPNet.ESNet(Nc, Nt, Nf)
        net = Constraint.Spectral_Normalization(net)

    elif algorithm == "CNNBIGRU":
        net = CNNBIGRU.ESNet(Nc, Nt, Nf)
        net = Constraint.Spectral_Normalization(net)
    elif algorithm == "CNNAttentionGRU":
        net =CNNAttentionGRU.ESNet(Nc,Nt,Nf)
        net = Constraint.Spectral_Normalization(net)
    elif algorithm == "CNNAttentionMLP":
        net =CNNAttentionMLP.ESNet(Nc, Nt, Nf)
        net = Constraint.Spectral_Normalization(net)
    elif algorithm == "CACAM":
        net = CACAM.ESNet(Nc, Nt, Nf)
        net = Constraint.Spectral_Normalization(net)
    elif algorithm == "DDGCNN":
        bz = config[algorithm]["bz"]
        norm = config[algorithm]["norm"]
        act = config[algorithm]["act"]
        trans_class = config[algorithm]["trans_class"]
        n_filters = config[algorithm]["n_filters"]
        net = DDGCNN.DenseDDGCNN([bz, Nt, Nc], k_adj=3, num_out=n_filters, dropout=0.5, n_blocks=3, nclass=Nf,
                                 bias=False, norm=norm, act=act, trans_class=trans_class, device=devices)

    net = net.to(devices)

    if algorithm == 'SSVEPNet' or algorithm=='CNNBIGRU' or algorithm=='CNNAttentionGRU' or algorithm=='CNNAttentionMLP' or algorithm=='CACAM' or algorithm=='CACAMNew':
        stimulus_type = str(config[algorithm]["stimulus_type"])
        criterion = LossFunction.CELoss_Marginal_Smooth(Nf, stimulus_type=stimulus_type)
    elif algorithm=='TFFFormer':
        criterion = LabelSmoothingLoss()
    else:
        criterion = nn.CrossEntropyLoss(reduction="none")

    if algorithm == "SSVEPformer" or algorithm == 'TFFBformer' or algorithm == 'TFformer' or algorithm == 'SSVEPformer2' or algorithm == 'SSVEPformer3' or algorithm == 'TFformerAtt' or algorithm=='FBTFformer3':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)



    return net, criterion, optimizer
