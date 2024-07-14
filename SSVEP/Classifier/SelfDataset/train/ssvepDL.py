import torch
import torch.nn as nn
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import sys

sys.path.append('../')
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import time

import torch.nn.functional as F

from model.SCUJJ import SCU
from utils import get_accuracy
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=60, help='number of epochs of training')
parser.add_argument('--lr', type=float, default=0.001, help='adam: learning rate')
parser.add_argument('--dropout_level', type=float, default=0.55, help='dropout level')
parser.add_argument('--w_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--seed_n', type=int, default=74, help='seed number')
opt = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(opt.seed_n)
torch.cuda.manual_seed(opt.seed_n)

num_classes = 40


def train_SCU(X_train, y_train):
    train_input = torch.from_numpy(X_train).float()
    train_label = torch.from_numpy(y_train)

    trainset = TensorDataset(train_input, train_label)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    cnn = SCU(opt, num_classes=num_classes).to(device)
    cnn.train()

    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=opt.lr, weight_decay=opt.w_decay)

    for epoch in range(opt.n_epochs):
        cumulative_accuracy = 0
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{opt.n_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = cnn(inputs)
            labels = labels.view(-1).long()

            loss = ce_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            cumulative_accuracy += (predicted == labels).sum().item() / labels.size(0)

        print(f"Epoch {epoch+1}/{opt.n_epochs}, Training Accuracy: {cumulative_accuracy / len(trainloader) * 100:.2f}%")

    return cnn


def test_SCU(cnn, X_test, y_test):
    test_input = torch.from_numpy(X_test).float()
    test_label = torch.from_numpy(y_test)

    testset = TensorDataset(test_input, test_label)
    testloader = DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    cnn.eval()
    test_cumulative_accuracy = 0

    for inputs, labels in tqdm(testloader, desc="Testing", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        test_outputs = cnn(inputs)
        _, test_predicted = torch.max(test_outputs, 1)

        test_acc = get_accuracy(labels, test_predicted)
        test_cumulative_accuracy += test_acc

    torch.save(cnn.state_dict(), f'../save/SCUJJ_{test_cumulative_accuracy / len(testloader) * 100:.2f}.pkl')
    return test_cumulative_accuracy, len(testloader)


def eval(cnn, X_test, y_test):
    random_index = np.random.randint(len(X_test))
    inputs = X_test[random_index]
    inputs = torch.from_numpy(inputs).float().to(device)

    with torch.no_grad():
        inputs = inputs.unsqueeze(0)
        outputs = cnn(inputs)
        _, predicted = torch.max(outputs, 1)
        predicted_label = predicted.item()

    true_label = y_test[random_index]
    print("Predicted Label:", predicted_label)
    print("True Label:", true_label)


if __name__ == "__main__":
    # EEGdata = np.load(r'../3x14-3x16-3x18-3x20/cyj/cyj_pySSVEP_3000.npy')
    # EEGlabel = np.load(r'../3x14-3x16-3x18-3x20/cyj/cyj_pySSVEP_3000_labels.npy')
    EEGdata = np.load(r'E:\02project\BCI-Github\DL_Classifier\s1mat_seq.npy')
    EEGlabel = np.load(r'E:\02project\BCI-Github\DL_Classifier\s1mat_seq_labels.npy')
    print(EEGlabel)
    EEGdata,EEGlabel=shuffle(EEGdata,EEGlabel,random_state=1337)
    X_train, X_test, y_train, y_test = train_test_split(EEGdata, EEGlabel, test_size=0.2)

    cnn = train_SCU(X_train, y_train)
    test_cumulative_accuracy, ntest = test_SCU(cnn, X_test, y_test)
    print(f"Test Accuracy: {test_cumulative_accuracy / ntest * 100:.2f}%")

    cnn.eval()
    for i in range(10):
        print(f"round {i}:")
        eval(cnn, X_test, y_test)
