import numpy as np
import matplotlib.pyplot as plt

def load_data():
    X = np.load("data/X_part1.npy")
    X_val = np.load("data/X_val_part1.npy")
    y_val = np.load("data/y_val_part1.npy")
    return X, X_val, y_val

def load_data_multi():
    X = np.load("data/X_part2.npy")
    X_val = np.load("data/X_val_part2.npy")
    y_val = np.load("data/y_val_part2.npy")
    return X, X_val, y_val


def multivariate_gaussian(X, mu, var):
    """
    计算样本X在多元高斯分布下的概率密度函数，
    分布参数为mu和var。如果var是矩阵，则视为协方差矩阵。
    如果var是向量，则视为每个维度的方差值（对角协方差矩阵）。
    """
    
    k = len(mu)
    
    if var.ndim == 1:
        var = np.diag(var)
        
    X = X - mu
    p = (2* np.pi)**(-k/2) * np.linalg.det(var)**(-0.5) * \
        np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1))
    
    return p
        
def visualize_fit(X, mu, var):
    """
    此可视化展示高斯分布的概率密度函数。
    每个样本有一个位置(x1, x2)，由其特征值决定。
    """
    
    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariate_gaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, var)
    Z = Z.reshape(X1.shape)

    plt.plot(X[:, 0], X[:, 1], 'bx')

    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, levels=10**(np.arange(-20., 1, 3)), linewidths=1)
        
    # 设置标题
    plt.title("拟合数据集的分布的高斯轮廓")
    # 设置y轴标签
    plt.ylabel('吞吐量 (mb/s)')
    # 设置x轴标签
    plt.xlabel('延迟 (ms)')