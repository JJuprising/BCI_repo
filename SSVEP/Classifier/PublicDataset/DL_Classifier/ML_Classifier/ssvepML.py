import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 加载数据
data = np.load(r'E:\02project\BCI-Github\DL_Classifier\s1mat_4dim.npy')  # (9, 1500, 40, 6)

# 检查数据形状
print("Data Shape:", data.shape)  # 应该是 (9, 1500, 40, 6)

# 生成标签
# 由于每个区块有 40 次试验，6 个区块，因此总共有 240 个标签
num_trials_per_block = data.shape[2]  # 每个区块 40 次试验
num_blocks = data.shape[3]  # 6 个区块
total_trials = num_trials_per_block * num_blocks  # 总共 240 次试验

# 创建标签（这里假设每个试验有对应的编号）
labels = np.tile(np.arange(num_trials_per_block), num_blocks)

# 确保标签长度与试验数一致
assert len(labels) == total_trials, "标签的数量不匹配"

# 数据处理
# 平均通道数据以得到每个试验的特征
X = np.mean(data, axis=0)  # 从(9, 1500, 40, 6) 到 (1500, 40, 6)

# 平坦化数据，使其符合模型输入要求
X = X.reshape(X.shape[0], -1).T  # (1500, 240)

# 确保数据形状与标签数量一致
# assert X.shape[1] == total_trials, "数据展平后与试验数不匹配"

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# 定义分类器
classifiers = {
    "SVM": SVC(kernel='linear', C=1.0, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

acc = {}

# 训练和评估模型
for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)

    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)

    # 计算评估指标
    accuracy_val = accuracy_score(y_val, y_val_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred, average='weighted')
    recall_test = recall_score(y_test, y_test_pred, average='weighted')
    f1_test = f1_score(y_test, y_test_pred, average='weighted')
    confusion_mat = confusion_matrix(y_test, y_test_pred)

    # 打印结果
    acc[clf_name] = accuracy_test
    print("Classifier:", clf_name)
    print("Validation Accuracy:", accuracy_val)
    print("Test Accuracy:", accuracy_test)
    print("Test Precision:", precision_test)
    print("Test Recall:", recall_test)
    print("Test F1-score:", f1_test)
    print("Confusion Matrix:\n", confusion_mat)
    print("\n")

print("Test Accuracy for all classifiers:", acc)
