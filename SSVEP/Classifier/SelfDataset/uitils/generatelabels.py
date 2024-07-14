import numpy as np

data=np.load('../3x14-3x16-3x18-3x20/cyj/cyj_pySSVEP_3000.npy')
print(data.shape) # (168, 2, 3000)
trials=int(data.shape[0])
# labels=np.zeros()
# 定义循环的标签
labels_cycle = np.array([0, 1, 2, 3])

# 将每个标签重复三次
labels_repeated = np.repeat(labels_cycle, 3)

# 重复该序列 trials // 4 次（除以 4 得到的结果）
labels = np.tile(labels_repeated, trials // 12)[:trials]

print(labels)
print(labels.shape)
np.save('../3x14-3x16-3x18-3x20/cyj/cyj_pySSVEP_3000_labels.npy', labels)