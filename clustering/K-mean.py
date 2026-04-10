# 主要用到的包
import pandas as pd #pandas 用来做表格 
import numpy as np #numpy 生成和处理数据矩阵
import matplotlib.pyplot as plt #画图
from sklearn.cluter import KMeans #核心K-means包

##### 以下是核心代码
X = df[feature_cols].values
##拟合不同 K
models = []
for k in range(2,10):
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    km.fit(X)
    models.append(km)
##比较 inertia，选 K
inertias = [km.inertia_ for km in models]
plt.plot(range(2,10), inertias, 'o-')

##最终模型
best_km = KMeans(n_clusters=chosen_k, n_init=20, random_state=42)
best_km.fit(X)
df['cluster'] = best_km.labels_
centers = best_km.cluster_centers_