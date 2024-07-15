# import csv
# import os
# import torch
# import time

# from matplotlib import pyplot as plt

# from etc.global_config import config

# classes = config['classes']  # 数据集
# if classes == 12:
#     ws = config["data_param_12"]["ws"]
#     Fs = config["data_param_12"]["Fs"]
# elif classes == 40:
#     ws = config["data_param_40"]["ws"]
#     Fs = config["data_param_40"]["Fs"]


# # 定义训练函数
# def train_on_batch(subject, num_epochs, train_iter, test_iter, optimizer, criterion, net, device, lr_jitter=False):
#     # 配置和参数
#     algorithm = config['algorithm']
#     width = config['KANformer']['width']

#     if algorithm == "DDGCNN":
#         lr_decay_rate = config[algorithm]['lr_decay_rate']
#         optim_patience = config[algorithm]['optim_patience']
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=lr_decay_rate,
#                                                                patience=optim_patience, verbose=True, eps=1e-08)
#     else:
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_iter),
#                                                                eta_min=5e-6)

#     best_val_acc = 0.0

#     # 设置结果保存路径
#     if algorithm == 'KANformer':
#         dir_path = f'../Result/classes_{classes}/{algorithm}/{str(width)}'
#     else:
#         dir_path = f'../Result/classes_{classes}/{algorithm}'

#     os.makedirs(dir_path, exist_ok=True)  # 确保目录存在

#     # 创建 CSV 文件
#     # csv_path = f'{dir_path}/subject_{subject}_ws({ws}s)_UD({config["train_param"]["UD"]}).csv'

#     # # 初始化训练信息列表
#     # train_messages = []

#     # 用于存储每个 epoch 的训练和验证准确度及损失
#     train_accuracies = []
#     train_losses = []
#     val_accuracies = []
#     val_losses = []

#     # 训练循环
#     for epoch in range(num_epochs):
#         net.train()
#         sum_loss = 0.0
#         sum_acc = 0.0

#         # 遍历训练迭代器
#         for batch_idx, data in enumerate(train_iter):
#             # 处理数据
#             if algorithm == "ConvCA":
#                 X, temp, y = data
#                 X = X.type(torch.FloatTensor).to(device)
#                 temp = temp.type(torch.FloatTensor).to(device)
#                 y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
#                 y_hat = net(X, temp)
#             else:
#                 X, y = data
#                 X = X.type(torch.FloatTensor).to(device)
#                 y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
#                 y_hat = net(X)

#             # 训练
#             loss = criterion(y_hat, y).sum()
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # if lr_jitter and algorithm != "DDGCNN":
#             #     scheduler.step()

#             sum_loss += loss.item() / y.shape[0]
#             sum_acc += (y == y_hat.argmax(dim=-1)).float().mean()

#         # 计算训练损失和准确度
#         train_loss = sum_loss / len(train_iter)
#         train_acc = (sum_acc / len(train_iter)).item()
#         # 计算平均训练准确度和损失
#         train_accuracies.append(train_acc)
#         train_losses.append(train_loss)

#         if lr_jitter and algorithm != "DDGCNN":
#             scheduler.step()
#         sum_acc = 0.0
#         sum_loss = 0.0
#         # 测试/验证过程
#         if 1:  # 每个epoch都验证
#             net.eval()

#             for data in test_iter:
#                 if algorithm == "ConvCA":
#                     X, temp, y = data
#                     X = X.type(torch.FloatTensor).to(device)
#                     temp = temp.type(torch.FloatTensor).to(device)
#                     y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
#                     y_hat = net(X, temp)
#                 else:
#                     X, y = data
#                     X = X.type(torch.FloatTensor).to(device)
#                     y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
#                     y_hat = net(X)

#                 sum_loss += criterion(y_hat, y).sum().item()
#                 sum_acc += (y == y_hat.argmax(dim=-1)).float().mean()

#             val_acc = (sum_acc / len(test_iter))
#             val_loss = (sum_loss / len(test_iter))
#             # 计算平均验证准确度和损失
#             val_accuracies.append(val_acc.item())
#             val_losses.append(val_loss)
#             # 保存最佳模型
#             if val_acc > best_val_acc:
#                 best_val_acc = val_acc
#                 # if algorithm == 'KANformer':
#                 #     torch.save(net,
#                 #                f'{dir_path}/best_Weights_ws({config["data_param_12"]["ws"]}s)_UD({config["train_param"]["UD"]})_width({width}).pkl')
#                 # else:
#                 #     torch.save(net,
#                 #                f'{dir_path}/best_Weights_ws({config["data_param_12"]["ws"]}s)_UD({config["train_param"]["UD"]}).pkl')

#             if epoch == num_epochs - 1 or (epoch + 1) % 10 == 0:
#                 print(f"epoch {epoch + 1}, val_acc = {val_acc:.3f},val_loss={val_loss:.3}")
#                 if lr_jitter and algorithm != "DDGCNN":
#                     scheduler.step()
#         # 在训练结束后，将训练信息保存为 CSV 文件
#         # train_message_csv_path = f'{csv_path}_train_messages.csv'
#         # with open(train_message_csv_path, 'w', newline='') as train_csvfile:
#         #     train_fieldnames = ['epoch', 'train_loss', 'train_acc']
#         #     train_writer = csv.DictWriter(train_csvfile, fieldnames=train_fieldnames)
#         #     train_writer.writeheader()

#         #     for msg in train_messages:
#         #         train_writer.writerow(msg)

#     # 绘制训练和验证的准确度与损失曲线
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#     print(train_accuracies)
#     print(val_accuracies)
#     print(train_losses)
#     print(val_losses)
#     ax1.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
#     ax1.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
#     ax1.set_xlabel("Epoch")
#     ax1.set_ylabel("Accuracy")
#     ax1.set_title(f"Epoch vs Accuracy FinAcc:{val_accuracies[-1]:.2f}")
#     ax1.legend()

#     ax2.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='orange')
#     ax2.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red')
#     ax2.set_xlabel("Epoch")
#     ax2.set_ylabel("Loss")
#     ax2.set_title("Epoch vs Loss")
#     ax2.legend()


#     timestamp = time.time()
#     # 保存图表到文件
#     plot_path = os.path.join(dir_path, f'training_results{ws}s_{subject}_{timestamp}.png')
#     plt.savefig(plot_path)
#     plt.close()
#     # for i in range(epoch):
#     #     print(f"epoch{i},acc:{val_acc[i]},loss:{val_loss[i]}")

#     # 清理 GPU 缓存
#     torch.cuda.empty_cache()

#     return best_val_acc.cpu().data.item()


# 定义训练函数
import csv
import os
import torch
import time

from matplotlib import pyplot as plt

from etc.global_config import config

classes = config['classes']  # 数据集
if classes == 12:
    # ws = config["data_param_12"]["ws"]
    Fs = config["data_param_12"]["Fs"]
elif classes == 40:
    # ws = config["data_param_40"]["ws"]
    Fs = config["data_param_40"]["Fs"]


# 定义训练函数
def train_on_batch(subject, num_epochs,val_interval, train_iter, test_iter, optimizer, criterion, net, device,ws, lr_jitter):
    # 配置和参数
    algorithm = config['algorithm']
    width = config['KANformer']['width']

    if algorithm == "DDGCNN":
        lr_decay_rate = config[algorithm]['lr_decay_rate']
        optim_patience = config[algorithm]['optim_patience']
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=lr_decay_rate,
                                                               patience=optim_patience, verbose=True, eps=1e-08)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_iter),
                                                               eta_min=5e-6)

    best_val_acc = 0.0

    # 设置结果保存路径
    if algorithm == 'KANformer':
        dir_path = f'../Result/classes_{classes}/{algorithm}/{str(width)}'
    else:
        dir_path = f'../Result/classes_{classes}/{algorithm}'

    os.makedirs(dir_path, exist_ok=True)  # 确保目录存在

    # 创建 CSV 文件
    csv_path = f'{dir_path}/subject_{subject}_ws({ws}s)_UD({config["train_param"]["UD"]}).csv'

    # 初始化训练信息列表
    train_messages = []

    # 用于存储每个 epoch 的训练和验证准确度及损失
    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []

    # 训练循环
    for epoch in range(num_epochs):
        net.train()
        sum_loss = 0.0
        sum_acc = 0.0

        # 遍历训练迭代器
        for batch_idx, data in enumerate(train_iter):
            # 处理数据
            if algorithm == "ConvCA":
                X, temp, y = data
                X = X.type(torch.FloatTensor).to(device)
                temp = temp.type(torch.FloatTensor).to(device)
                y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                y_hat = net(X, temp)
            else:
                X, y = data
                X = X.type(torch.FloatTensor).to(device)
                y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                y_hat = net(X)

            # 训练
            loss = criterion(y_hat, y).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if lr_jitter and algorithm != "DDGCNN":
                scheduler.step()

            sum_loss += loss.item() / y.shape[0]
            sum_acc += (y == y_hat.argmax(dim=-1)).float().mean()

        # 计算训练损失和准确度
        train_loss = sum_loss / len(train_iter)
        train_acc = (sum_acc / len(train_iter)).item()
        # 计算平均训练准确度和损失
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)

        if lr_jitter and algorithm == "DDGCNN":
            scheduler.step(train_acc)
        sum_acc = 0.0
        sum_loss = 0.0
        # 测试/验证过程
        if 1:  # 每个epoch都验证
            net.eval()

            for data in test_iter:
                if algorithm == "ConvCA":
                    X, temp, y = data
                    X = X.type(torch.FloatTensor).to(device)
                    temp = temp.type(torch.FloatTensor).to(device)
                    y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                    y_hat = net(X, temp)
                else:
                    X, y = data
                    X = X.type(torch.FloatTensor).to(device)
                    y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                    y_hat = net(X)

                sum_loss += criterion(y_hat, y).sum().item()
                sum_acc += (y == y_hat.argmax(dim=-1)).float().mean()

            val_acc = (sum_acc / len(test_iter))
            val_loss = (sum_loss / len(test_iter))
            # 计算平均验证准确度和损失
            val_accuracies.append(val_acc.item())
            val_losses.append(val_loss)
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # if algorithm == 'KANformer':
                #     torch.save(net,
                #                f'{dir_path}/best_Weights_ws({config["data_param_12"]["ws"]}s)_UD({config["train_param"]["UD"]})_width({width}).pkl')
                # else:
                #     torch.save(net,
                #                f'{dir_path}/best_Weights_ws({config["data_param_12"]["ws"]}s)_UD({config["train_param"]["UD"]}).pkl')

            if epoch == num_epochs - 1 or (epoch + 1) % 10 == 0:
                print(f"epoch {epoch + 1}, val_acc = {val_acc:.3f}")

        # 在训练结束后，将训练信息保存为 CSV 文件
        train_message_csv_path = f'{csv_path}_train_messages.csv'
        with open(train_message_csv_path, 'w', newline='') as train_csvfile:
            train_fieldnames = ['epoch', 'train_loss', 'train_acc']
            train_writer = csv.DictWriter(train_csvfile, fieldnames=train_fieldnames)
            train_writer.writeheader()

            for msg in train_messages:
                train_writer.writerow(msg)

    # 绘制训练和验证的准确度与损失曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # print(train_accuracies)
    # print(val_accuracies)
    # print(train_losses)
    # print(val_losses)
    ax1.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    ax1.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Epoch vs Accuracy FinAcc:{val_accuracies[-1]:.2f}")
    ax1.legend()

    ax2.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='orange')
    ax2.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Epoch vs Loss ")
    ax2.legend()


    timestamp = time.time()
    # 保存图表到文件
    plot_path = os.path.join(dir_path, f'training_results{ws}s_{subject}_{timestamp}.png')
    plt.savefig(plot_path)
    plt.close()
    # 清理 GPU 缓存
    torch.cuda.empty_cache()

    return best_val_acc.cpu().data.item()

