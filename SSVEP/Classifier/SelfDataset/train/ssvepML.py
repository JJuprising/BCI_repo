import numpy as np
from brainflow import DataFilter, FilterTypes, DetrendOperations
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# 加载数据和标签
from sklearn.tree import DecisionTreeClassifier
acc = {}
data = np.load(r'E:\02project\BCI-Github\DL_Classifier\s1mat_seq.npy')  # (通道，采样点,trails,blocks)
labels = np.load(r'E:\02project\BCI-Github\DL_Classifier\s1mat_seq_labels.npy')

# data=np.load(r'E:\02project\BCI-Github\BCI-UAV\BCIcode\quickssvep-master\pythondata\3x14-3x16-3x18-3x20\cyj\cyj_pySSVEP_3000.npy')
# labels=np.load(r'E:\02project\BCI-Github\BCI-UAV\BCIcode\quickssvep-master\pythondata\3x14-3x16-3x18-3x20\cyj\cyj_pySSVEP_3000_labels.npy')
# data=data[:40,:]
# labels=labels[:40]

print(labels)
trials = data.shape[0]
print(data.shape)
print(labels.shape)
# 通道平均
X = np.mean(data, axis=1)


# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 划分验证集

classifiers = {
    "SVM": SVC(kernel='linear', C=1.0, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

for clf_name, clf in classifiers.items():
    # 训练分类器
    clf.fit(X_train, y_train)

    # 预测验证集和测试集
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)

    # 计算评估指标
    accuracy_val = accuracy_score(y_val, y_val_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred, average='weighted')
    recall_test = recall_score(y_test, y_test_pred, average='weighted')
    f1_test = f1_score(y_test, y_test_pred, average='weighted')
    confusion_mat = confusion_matrix(y_test, y_test_pred)

    # 打印评估结果
    acc[clf_name] = accuracy_test
    print("Classifier:", clf_name)
    print("Validation Accuracy:", accuracy_val)
    print("Test Accuracy:", accuracy_test)
    print("Test Precision:", precision_test)
    print("Test Recall:", recall_test)
    print("Test F1-score:", f1_test)
    print("Confusion Matrix:\n", confusion_mat)
    print("\n")

print(acc)
