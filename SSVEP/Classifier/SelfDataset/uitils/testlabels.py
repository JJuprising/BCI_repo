
# 检测有效数据段
# data = np.loadtxt('./data/BrainFlow-RAW_2024-01-18_23-14-43_0.csv').T
import numpy as np

data=np.loadtxt(r'E:\02project\BCI-Github\BCI-UAV\BCIcode\quickssvep-master\pythondata\3x14-3x16-3x18-3x20\cyj\BrainFlow-RAW_2024-04-09_22-25-02_0.csv').T
data=np.loadtxt(r'E:\02project\BCI-Github\BCI-UAV\BCIcode\quickssvep-master\pythondata\3x14-3x16-3x18-3x20\cyj\BrainFlow-RAW_2024-04-09_21-55-14_0.csv').T
print(data.shape)
count=0
labels = data[-2]
eeg_data = data[[2,3]]
indexl=0
indexr=0
for index in range(len(labels)):
    label = labels[index]
    if label != 0:
        indexl=indexr
        indexr=index
        print(index,indexr-indexl,label)
        count+=1

print(count)
