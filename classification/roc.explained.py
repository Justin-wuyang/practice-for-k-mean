"""
roc_explained.py

功能：
1. 生成一个模拟的二分类数据集
2. 用训练集训练 Logistic Regression（逻辑回归）模型
3. 在测试集上预测
4. 计算 ROC curve、AUC、confusion matrix
5. 计算 sensitivity / specificity / accuracy
6. 画出 ROC 曲线和混淆矩阵

这份文件适合作为“学习版”脚本保存到 GitHub。
每一段代码都附了中文解释，方便以后复习。
"""

# =========================
# 1. 导入需要的库
# =========================

# matplotlib 用来画图
import matplotlib.pyplot as plt

# sklearn.metrics 里放的是模型评估工具
from sklearn.metrics import (
    roc_curve,               # 计算 ROC 曲线需要的 FPR / TPR / threshold
    roc_auc_score,           # 计算 AUC
    confusion_matrix,        # 计算混淆矩阵
    ConfusionMatrixDisplay   # 把混淆矩阵画出来
)

# train_test_split 用来把数据拆成训练集和测试集
from sklearn.model_selection import train_test_split

# LogisticRegression 是一个经典的二分类模型
from sklearn.linear_model import LogisticRegression

# make_classification 用来生成一个“练习用”的二分类模拟数据
from sklearn.datasets import make_classification


# =========================
# 2. 生成模拟数据
# =========================

# 这里生成一个二分类数据集：
# n_samples=1000 表示一共有 1000 个样本
# n_features=20 表示每个样本有 20 个特征
# n_classes=2 表示是二分类问题（0 和 1）
# random_state=101 用来固定随机结果，保证你每次运行结果一致
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    random_state=101
)

# 这里的 X 是“特征矩阵”
# 可以理解成：每一行是一个人，每一列是一个特征
#
# 这里的 y 是“标签”
# 可以理解成：每个人真实属于 0 类还是 1 类


# =========================
# 3. 划分训练集和测试集
# =========================

# 把数据分成训练集和测试集
# test_size=0.2 表示 20% 拿去测试，80% 用来训练
# random_state=101 继续固定随机结果
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=101
)

# 解释：
# X_train：训练集的特征
# X_test：测试集的特征
# y_train：训练集的真实标签
# y_test：测试集的真实标签


# =========================
# 4. 建立并训练逻辑回归模型
# =========================

# 创建一个逻辑回归模型对象
# max_iter=1000 是为了防止有时默认迭代次数不够，模型不收敛
clf = LogisticRegression(max_iter=1000)

# 用训练集训练模型
# 这一行就是“让模型从训练数据中学习”
clf.fit(X_train, y_train)


# =========================
# 5. 用模型做预测
# =========================

# predict_proba(X_test) 会输出每个测试样本属于各类别的概率
# 二分类时，输出通常是两列：
# 第 1 列：属于 0 类的概率
# 第 2 列：属于 1 类的概率
#
# [:, 1] 的意思是：
# 取“所有行”的“第 2 列”
# 也就是取每个样本“属于正类（1）”的概率
y_scores = clf.predict_proba(X_test)[:, 1]

# predict(X_test) 会直接输出最终预测类别（0 或 1）
# 它会根据默认阈值（通常是 0.5）来判断：
# 概率 > 0.5 -> 预测为 1
# 概率 <= 0.5 -> 预测为 0
y_pred = clf.predict(X_test)

# 解释：
# y_test   = 测试集真实答案
# y_scores = 模型给出的“属于 1 类的概率”
# y_pred   = 模型最终给出的 0/1 分类结果


# =========================
# 6. 计算 ROC 曲线和 AUC
# =========================

# roc_curve 需要两个输入：
# 1. y_test   -> 真实标签
# 2. y_scores -> 模型给出的概率分数
#
# 输出：
# fpr        -> False Positive Rate（假阳性率）
# tpr        -> True Positive Rate（真阳性率 / sensitivity）
# thresholds -> 不同的分类阈值
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# AUC = ROC 曲线下面的面积
# AUC 越接近 1，模型区分正负类的能力越强
auc = roc_auc_score(y_test, y_scores)


# =========================
# 7. 计算混淆矩阵
# =========================

# confusion_matrix 比较“真实标签”和“最终预测标签”
conf_matrix = confusion_matrix(y_test, y_pred)

# 对于二分类，混淆矩阵是一个 2x2 表
# ravel() 会把它按顺序展开成 4 个数
TN, FP, FN, TP = conf_matrix.ravel()

# 解释：
# TN = True Negative   真实为 0，预测也为 0
# FP = False Positive  真实为 0，预测成 1
# FN = False Negative  真实为 1，预测成 0
# TP = True Positive   真实为 1，预测也为 1


# =========================
# 8. 计算常见指标
# =========================

# sensitivity = 真阳性率 = recall
# 所有真实阳性中，被正确识别出来的比例
sensitivity = TP / (TP + FN)

# specificity = 真阴性率
# 所有真实阴性中，被正确识别出来的比例
specificity = TN / (TN + FP)

# accuracy = 总体正确率
accuracy = (TP + TN) / (TP + TN + FP + FN)


# =========================
# 9. 打印结果
# =========================

print("===== Model Evaluation Metrics =====")
print(f"AUC: {auc:.2f}")
print(f"Sensitivity (True Positive Rate): {sensitivity:.2f}")
print(f"Specificity (True Negative Rate): {specificity:.2f}")
print(f"Accuracy: {accuracy:.2f}")

print("\n===== Confusion Matrix =====")
print(conf_matrix)


# =========================
# 10. 绘制 ROC 曲线
# =========================

plt.figure(figsize=(8, 6))

# 画 ROC 曲线
plt.plot(
    fpr, tpr,
    label=f'ROC Curve (AUC = {auc:.2f})',
    linewidth=2
)

# 画一条随机分类器参考线（对角线）
plt.plot(
    [0, 1], [0, 1],
    linestyle='--',
    label='Random Classifier'
)

plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('Receiver Operating Characteristic Curve', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# =========================
# 11. 绘制混淆矩阵
# =========================

disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix,
    display_labels=clf.classes_
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()