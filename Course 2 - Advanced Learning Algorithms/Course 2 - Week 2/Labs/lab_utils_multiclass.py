# C2_W1 工具函数
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 绘制多类别训练点
def plot_mc_data(X, y, class_labels=None, legend=False,size=40):
    classes = np.unique(y)
    for i in classes:
        label = class_labels[i] if class_labels else "类别 {}".format(i)
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1],  cmap=plt.cm.Paired,
                    edgecolor='black', s=size, label=label)
    if legend: plt.legend()
        

#绘制多类别分类决策边界
# 此版本处理非向量预测（添加了对点的for循环）
def plot_cat_decision_boundary(X,predict , class_labels=None, legend=False, vector=True):

    # 创建网格点用于绘图
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = max(x_max-x_min, y_max-y_min)/200
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    points = np.c_[xx.ravel(), yy.ravel()]
    print("点", points.shape)
    print("xx.shape", xx.shape)

    #对网格中的每个点进行预测
    if vector:
        Z = predict(points)
    else:
        Z = np.zeros((len(points),))
        for i in range(len(points)):
            Z[i] = predict(points[i].reshape(1,2))
    Z = Z.reshape(xx.shape)

    #等高线图突出显示值之间的边界 - 在本例中为类别
    plt.figure()
    plt.contour(xx, yy, Z, colors='g') 
    plt.axis('tight')