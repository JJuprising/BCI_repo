import numpy as np
import matplotlib.pyplot as plt

# 定义模型名称和对应的准确度及标准差
models = ['CCNN', 'ConvCA', 'EEGNet', 'FBtCNN', 'DDGCNN', 'SSVEPformer']
accuracies = [88.00, 73.08, 77.67, 82.00, 80.92, 90.08]  # 以示例数据为准，需替换为实际的模型准确度数据
std_devs = [19.95, 15.02, 28.93, 20.55, 23.60, 22.58]  # 以示例数据为准，需替换为实际的标准差数据

# 生成颜色列表，保证每个柱子的颜色不同
colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

# 绘制柱状图
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=colors)

# 在每个柱子上方添加标准差数值
for bar, std_dev in zip(bars, std_devs):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f} ± {std_dev:.2f}', ha='center', va='bottom', fontsize=10)

# 添加标题和标签
# plt.title('Comparison of Model Accuracies')
plt.xlabel('Methods')
plt.ylabel('Accuracy')

# 显示柱状图
plt.xticks(rotation=45)
plt.ylim(30, 100)  # 设置y轴范围
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()  # 自动调整布局，防止标签重叠
plt.show()
