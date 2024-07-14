import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 文件夹路径
folder_path = "./"  # 替换为你的文件夹路径

# 被试的数量
num_subjects = 3

# 存储每个被试的最后一个验证准确率
last_epoch_accuracies = []
subject_ids = []
algorithm='TFformerAtt'
# 遍历所有文件
for subject_id in range(1, num_subjects + 1):
    # 生成文件名
    file_name = f"../Result/classes_40/{algorithm}/subject_{subject_id}_ws(1.0s)_UD(1).csv"
    file_path = os.path.join(folder_path, file_name)

    # 读取数据
    data = pd.read_csv(file_path, skipinitialspace=True)  # 移除分隔符后的空格

    # 获取最后一个 epoch 的验证准确率
    last_val_acc = data['val_acc'].iloc[-1]

    # 存储验证准确率和被试 ID
    last_epoch_accuracies.append(last_val_acc)
    subject_ids.append(subject_id)

# 计算平均准确率和标准差
mean_accuracy = np.mean(last_epoch_accuracies)
std_deviation = np.std(last_epoch_accuracies)

# 绘制柱状图
x = np.arange(len(subject_ids))  # x轴的刻度
width = 0.5  # 柱子的宽度

fig, ax = plt.subplots(figsize=(12, 8))  # 扩大画幅

# 绘制每个被试的验证准确率柱状图
bars = ax.bar(x, last_epoch_accuracies, width, label='Accuracy')

# 在柱体上方显示数值
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 4), va='bottom', ha='center')

# 添加平均准确率和标准差的线和文本
ax.axhline(mean_accuracy, color='red', linestyle='--', label=f'Mean = {round(mean_accuracy, 4)}')
ax.axhline(mean_accuracy + std_deviation, color='orange', linestyle=':', label=f'Std = {round(std_deviation, 4)}')

# 设置标签和标题
ax.set_xlabel('Subject ID')
ax.set_ylabel('Validation Accuracy')
ax.set_title('Validation Accuracy for Last Epochs of Subjects')
ax.set_xticks(x)  # 设置x轴刻度
ax.set_xticklabels(subject_ids)  # 设置x轴刻度标签
ax.legend()
plt.savefig(f'../Result/classes_40/{algorithm}/result.png')
# 显示图表
plt.show()
