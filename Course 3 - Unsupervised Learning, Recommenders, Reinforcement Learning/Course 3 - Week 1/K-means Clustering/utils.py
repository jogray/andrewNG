import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

def load_data():
    X = np.load("data/ex7_X.npy")
    return X

def draw_line(p1, p2, style="-k", linewidth=1):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], style, linewidth=linewidth)

def plot_data_points(X, idx):
    # 定义颜色映射以匹配笔记本中的图1
    cmap = ListedColormap(["red", "green", "blue"])
    c = cmap(idx)
    
    # 绘制X中的数据点，使idx中具有相同索引分配的点具有相同的颜色
    plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors=c, linewidth=0.1, alpha=0.7)

def plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i):
    # 绘制样本
    plot_data_points(X, idx)
    
    # 将质心绘制为黑色'x'标记
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='k', linewidths=3)
    
    # 用线绘制质心的历史轨迹
    for j in range(centroids.shape[0]):
        draw_line(centroids[j, :], previous_centroids[j, :])
    
    plt.title("迭代次数 %d" %i)


def plot_kMeans_RGB(X, centroids, idx, K):
    # 在3D空间中绘制颜色和质心
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(*X.T*255, zdir='z', depthshade=False, s=.3, c=X)
    ax.scatter(*centroids.T*255, zdir='z', depthshade=False, s=500, c='red', marker='x', lw=3)
    ax.set_xlabel('R值 - 红色')
    ax.set_ylabel('G值 - 绿色')
    ax.set_zlabel('B值 - 蓝色')
    ax.yaxis.pane.set_facecolor((0., 0., 0., .2))
    ax.set_title("原始颜色及其颜色聚类的质心")
    plt.show()


def show_centroid_colors(centroids):
    palette = np.expand_dims(centroids, axis=0)
    num = np.arange(0,len(centroids))
    plt.figure(figsize=(16, 16))
    plt.xticks(num)
    plt.yticks([])
    plt.imshow(palette)
