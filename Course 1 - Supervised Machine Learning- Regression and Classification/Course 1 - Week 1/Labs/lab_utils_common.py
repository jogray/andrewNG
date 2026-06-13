""" 
lab_utils_common.py
    所有可选实验室通用的函数，课程1，第2周
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./deeplearning.mplstyle')
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0';
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]
dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')


##########################################################
# 回归例程
##########################################################

#计算代价的函数
def compute_cost_matrix(X, y, w, b, verbose=False):
    """
    计算线性回归的梯度
     参数:
      X (ndarray (m,n)): 数据，m个样本，n个特征
      y (ndarray (m,)) : 目标值
      w (ndarray (n,)) : 模型参数
      b (scalar)       : 模型参数
      verbose : (Boolean) 如果为True，打印中间值f_wb
    返回
      cost: (scalar)
    """
    m = X.shape[0]

    # 计算所有样本的f_wb。
    f_wb = X @ w + b
    # 计算代价
    total_cost = (1/(2*m)) * np.sum((f_wb-y)**2)

    if verbose: print("f_wb:")
    if verbose: print(f_wb)

    return total_cost

def compute_gradient_matrix(X, y, w, b):
    """
    计算线性回归的梯度

    参数:
      X (ndarray (m,n)): 数据，m个样本，n个特征
      y (ndarray (m,)) : 目标值
      w (ndarray (n,)) : 模型参数
      b (scalar)       : 模型参数
    返回
      dj_dw (ndarray (n,1)): 代价相对于参数w的梯度。
      dj_db (scalar):        代价相对于参数b的梯度。

    """
    m,n = X.shape
    f_wb = X @ w + b
    e   = f_wb - y
    dj_dw  = (1/m) * (X.T @ e)
    dj_db  = (1/m) * np.sum(e)

    return dj_db,dj_dw


# 多变量compute_cost的循环版本
def compute_cost(X, y, w, b):
    """
    计算代价
    参数:
      X (ndarray (m,n)): 数据，m个样本，n个特征
      y (ndarray (m,)) : 目标值
      w (ndarray (n,)) : 模型参数
      b (scalar)       : 模型参数
    返回
      cost (scalar)    : 代价
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i],w) + b           #(n,)(n,)=scalar
        cost = cost + (f_wb_i - y[i])**2
    cost = cost/(2*m)
    return cost 

def compute_gradient(X, y, w, b):
    """
    计算线性回归的梯度
    参数:
      X (ndarray (m,n)): 数据，m个样本，n个特征
      y (ndarray (m,)) : 目标值
      w (ndarray (n,)) : 模型参数
      b (scalar)       : 模型参数
    返回
      dj_dw (ndarray Shape (n,)): 代价相对于参数w的梯度。
      dj_db (scalar):             代价相对于参数b的梯度。
    """
    m,n = X.shape           #(样本数量, 特征数量)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_db,dj_dw

