import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import os
os.environ["PATH"] += os.pathsep + 'D:\Program Files\graphviz\bin'
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve
from io import StringIO
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.model_selection import KFold
from IPython.display import Image
import pydotplus
import copy

import rasterio


#数据预处理阶段
#然后读取新数据集进行分析
data = pd.read_csv('train_0_to_49.csv', index_col=None)
# 检测是否有缺失数据
print(data.isnull().any())
# 数据的样例
print(data.head())
# 共38984个样本，每一个样本中包含12个特征
print(data.shape)
# 显示统计数据
print(data.describe())
#相关性分析，进行降维
corr = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm',
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.savefig('correlation_heatmap.png')
plt.show()

# 产生X, y，即特征值与目标值
target_name = 'label'
X = data.drop('label', axis=1)
y = data[target_name]
# 显示前5行数据
print(X.head())

# 定义 K 折交叉验证的折数
n_splits = 10

# 创建 KFold 对象
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=52)
# 定义训练轮数
num = 1
# 使用 KFold 进行交叉验证划分
for train_indices, val_indices in kfold.split(X):
    # 根据索引获取训练集和验证集
    X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
    y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]
    # 实例化
    dtree = tree.DecisionTreeClassifier(
        criterion='entropy',
        # max_depth=3, # 定义树的深度, 可以用来防止过拟合
        min_weight_fraction_leaf=0.01  # 定义叶子节点最少需要包含多少个样本(使用百分比表达), 防止过拟合
    )
    # 训练
    dtree = dtree.fit(X_train, y_train)
    # 指标计算
    dt_roc_auc = roc_auc_score(y_val, dtree.predict(X_val))
    print("决策树 AUC = %2.2f" % dt_roc_auc)
    print(classification_report(y_val, dtree.predict(X_val)))
    if num==1:
        dtree2 = tree.DecisionTreeClassifier(
            criterion='entropy',
            # max_depth=3, # 定义树的深度, 可以用来防止过拟合
            min_weight_fraction_leaf=0.01  # 定义叶子节点最少需要包含多少个样本(使用百分比表达), 防止过拟合
        )
        # 训练
        dtree2 = dtree2.fit(X_train, y_train)
        # 指标计算
        dt2_roc_auc = roc_auc_score(y_val, dtree2.predict(X_val))
        X_val_dt = X_val.copy()
        y_val_dt = y_val.copy()
    num+=1
    if dt2_roc_auc<dt_roc_auc:
        dtree2=copy.copy(dtree)
        X_val_dt = X_val.copy()
        y_val_dt = y_val.copy()
joblib.dump(dtree2, 'dtree_model.pkl')
# 需安装GraphViz和pydotplus进行决策树的可视化
# 特征向量
feature_names = data.columns[:-1]
# 文件缓存
dot_data = StringIO()
# 将决策树导入到dot中
export_graphviz(dtree2, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_names,class_names=['0','1'])
# 将生成的dot文件生成graph
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# 将结果存入到png文件中
graph.write_png('diabetes.png')
# 显示
Image(graph.create_png())
importances = dtree2.feature_importances_
# 获取特征名称
feat_names = data.drop(['label'],axis=1).columns
# 排序
indices = np.argsort(importances)[::-1]
# 绘图
plt.figure(figsize=(12,6))
plt.title("Feature importances by Decision Tree")
plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()







# 定义 K 折交叉验证的折数
n_splits = 10

# 创建 KFold 对象
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=52)
# 定义训练轮数
num = 1
# 使用 KFold 进行交叉验证划分
for train_indices, val_indices in kfold.split(X):
    # 根据索引获取训练集和验证集
    X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
    y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]
    #随机森林实例化
    rf = RandomForestClassifier(
        criterion='entropy',
        n_estimators=50,
        max_depth=None,  # 定义树的深度, 可以用来防止过拟合
        min_samples_split=10,  # 定义至少多少个样本的情况下才继续分叉
        # min_weight_fraction_leaf=0.02 # 定义叶子节点最少需要包含多少个样本(使用百分比表达), 防止过拟合
    )
    # 模型训练
    rf.fit(X_train, y_train)
    # 计算指标参数
    rf_roc_auc = roc_auc_score(y_val, rf.predict(X_val))
    print("随机森林 AUC = %2.2f" % rf_roc_auc)
    print(classification_report(y_val, rf.predict(X_val)))
    if num==1:
        rf2= RandomForestClassifier(
            criterion='entropy',
            n_estimators=50,
            max_depth=None,  # 定义树的深度, 可以用来防止过拟合
            min_samples_split=10,  # 定义至少多少个样本的情况下才继续分叉
            min_weight_fraction_leaf=0.02 # 定义叶子节点最少需要包含多少个样本(使用百分比表达), 防止过拟合
        )
        # 模型训练
        rf2.fit(X_train, y_train)
        rf2_roc_auc = roc_auc_score(y_val, rf2.predict(X_val))
        X_val_rf = X_val.copy()
        y_val_rf = y_val.copy()
    num += 1
    if rf2_roc_auc < rf_roc_auc:
        rf2 = copy.deepcopy(rf)
        X_val_rf = X_val.copy()
        y_val_rf = y_val.copy()
joblib.dump(rf2, 'rf_model.pkl')


'''
# Graphviz中未提供多棵树的绘制方法，所以我们遍历森林中的树，分别进行绘制
Estimators = rf2.estimators_
# 遍历
for index, model in enumerate(Estimators):
    # 文件缓存
    dot_data = StringIO()
    # 将决策树导入到dot_data中
    export_graphviz(model , out_file=dot_data,
                         feature_names=data.columns[1:],
                         class_names=['0','1'],
                         filled=True, rounded=True,
                         special_characters=True)
    # 从数据中生成graph
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # 将结果写入到png文件中
    graph.write_png('Rf{}.png'.format(index))
    # 绘制图像
    plt.figure(figsize = (20,20))
    plt.imshow(plt.imread('Rf{}.png'.format(index)))
    plt.axis('off')
'''
# 特征的重要程度
importances = rf2.feature_importances_
# 特征名称
feat_names = data.drop(['label'],axis=1).columns
# 排序
indices = np.argsort(importances)[::-1]
# 绘图
plt.figure(figsize=(12,6))
plt.title("Feature importances by RandomForest")
plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()




# ROC 图
from sklearn.metrics import roc_curve
# 计算ROC曲线
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_val_rf, rf2.predict_proba(X_val_rf)[:,1])
dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_val_dt, dtree2.predict_proba(X_val_dt)[:,1])
plt.figure()

# 随机森林 ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
# 决策树 ROC
plt.plot(dt_fpr, dt_tpr, label='Decision Tree (area = %0.2f)' % dt_roc_auc)
# 绘图
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()