"""
lab_utils_common
   包含所有本周实验通用的例程和变量定义。
   相比之下，特定的大型绘图例程将在单独的文件中，
   通常在使用的周中导入。
   这些文件将导入此文件
"""
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from ipywidgets import Output

np.set_printoptions(precision=2)

dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]
plt.style.use('./deeplearning.mplstyle')

def sigmoid(z):
    """
    计算z的sigmoid值

    参数
    ----------
    z : array_like
        任意大小的标量或numpy数组。

    返回
    -------
     g : array_like
         sigmoid(z)
    """
    z = np.clip( z, -500, 500 )           # 防止溢出
    g = 1.0/(1.0+np.exp(-z))

    return g

##########################################################
# 回归例程
##########################################################

def predict_logistic(X, w, b):
    """ 执行预测 """
    return sigmoid(X @ w + b)

def predict_linear(X, w, b):
    """ 执行预测 """
    return X @ w + b

def compute_cost_logistic(X, y, w, b, lambda_=0, safe=False):
    """
    使用逻辑损失计算代价，非矩阵版本

    参数:
      X (ndarray): Shape (m,n)  n特征样本矩阵
      y (ndarray): Shape (m,)   目标值
      w (ndarray): Shape (n,)   预测参数
      b (scalar):               预测参数
      lambda_ : (scalar, float) 控制正则化的程度，0 = 无正则化
      safe : (boolean)          True-选择下溢/溢出安全算法
    返回:
      cost (scalar): 代价
    """

    m,n = X.shape
    cost = 0.0
    for i in range(m):
        z_i    = np.dot(X[i],w) + b                                             #(n,)(n,) or (n,) ()
        if safe:  #避免溢出
            cost += -(y[i] * z_i ) + log_1pexp(z_i)
        else:
            f_wb_i = sigmoid(z_i)                                                   #(n,)
            cost  += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)       # scalar
    cost = cost/m

    reg_cost = 0
    if lambda_ != 0:
        for j in range(n):
            reg_cost += (w[j]**2)                                               # scalar
        reg_cost = (lambda_/(2*m))*reg_cost

    return cost + reg_cost


def log_1pexp(x, maximum=20):
    ''' 近似计算 log(1+exp^x)
        https://stats.stackexchange.com/questions/475589/numerical-computation-of-cross-entropy-in-practice
    参数:
    x   : (ndarray Shape (n,1) or (n,))  输入
    out : (ndarray Shape matches x)      输出 ~= np.log(1+exp(x))
    '''

    out  = np.zeros_like(x,dtype=float)
    i    = x <= maximum
    ni   = np.logical_not(i)

    out[i]  = np.log(1 + np.exp(x[i]))
    out[ni] = x[ni]
    return out


def compute_cost_matrix(X, y, w, b, logistic=False, lambda_=0, safe=True):
    """
    使用矩阵计算代价
    参数:
      X : (ndarray, Shape (m,n))          样本矩阵
      y : (ndarray  Shape (m,) or (m,1))  每个样本的目标值
      w : (ndarray  Shape (n,) or (n,1))  模型参数的值
      b : (scalar)                        模型参数的值
      verbose : (Boolean) 如果为True，打印中间值f_wb
    返回:
      total_cost: (scalar)                代价
    """
    m = X.shape[0]
    y = y.reshape(-1,1)             # ensure 2D
    w = w.reshape(-1,1)             # ensure 2D
    if logistic:
        if safe:  #safe from overflow
            z = X @ w + b                                                           #(m,n)(n,1)=(m,1)
            cost = -(y * z) + log_1pexp(z)
            cost = np.sum(cost)/m                                                   # (scalar)
        else:
            f    = sigmoid(X @ w + b)                                               # (m,n)(n,1) = (m,1)
            cost = (1/m)*(np.dot(-y.T, np.log(f)) - np.dot((1-y).T, np.log(1-f)))   # (1,m)(m,1) = (1,1)
            cost = cost[0,0]                                                        # scalar
    else:
        f    = X @ w + b                                                        # (m,n)(n,1) = (m,1)
        cost = (1/(2*m)) * np.sum((f - y)**2)                                   # scalar

    reg_cost = (lambda_/(2*m)) * np.sum(w**2)                                   # scalar

    total_cost = cost + reg_cost                                                # scalar

    return total_cost                                                           # scalar

def compute_gradient_matrix(X, y, w, b, logistic=False, lambda_=0):
    """
    使用矩阵计算梯度

    参数:
      X : (ndarray, Shape (m,n))          样本矩阵
      y : (ndarray  Shape (m,) or (m,1))  每个样本的目标值
      w : (ndarray  Shape (n,) or (n,1))  模型参数的值
      b : (scalar)                        模型参数的值
      logistic: (boolean)                 如果为false则为线性，如果为true则为逻辑
      lambda_:  (float)                   如果非零则应用正则化
    返回
      dj_dw: (array_like Shape (n,1))     代价相对于参数w的梯度
      dj_db: (scalar)                     代价相对于参数b的梯度
    """
    m = X.shape[0]
    y = y.reshape(-1,1)             # ensure 2D
    w = w.reshape(-1,1)             # ensure 2D

    f_wb  = sigmoid( X @ w + b ) if logistic else  X @ w + b      # (m,n)(n,1) = (m,1)
    err   = f_wb - y                                              # (m,1)
    dj_dw = (1/m) * (X.T @ err)                                   # (n,m)(m,1) = (n,1)
    dj_db = (1/m) * np.sum(err)                                   # scalar

    dj_dw += (lambda_/m) * w        # 正则化                  # (n,1)

    return dj_db, dj_dw                                           # scalar, (n,1)

def gradient_descent(X, y, w_in, b_in, alpha, num_iters, logistic=False, lambda_=0, verbose=True):
    """
    执行批量梯度下降来学习theta。通过执行
    num_iters次梯度下降步骤，学习率为alpha来更新theta

    参数:
      X (ndarray):    Shape (m,n)         样本矩阵
      y (ndarray):    Shape (m,) or (m,1) 每个样本的目标值
      w_in (ndarray): Shape (n,) or (n,1) 模型参数的初始值
      b_in (scalar):                      模型参数的初始值
      logistic: (boolean)                 如果为false则为线性，如果为true则为逻辑
      lambda_:  (float)                   如果非零则应用正则化
      alpha (float):                      学习率
      num_iters (int):                    运行梯度下降的迭代次数

    返回:
      w (ndarray): Shape (n,) or (n,1)    参数的更新值；匹配输入形状
      b (scalar):                         参数的更新值
    """
    # 一个数组，用于存储每次迭代的代价J和w，主要用于后续绘图
    J_history = []
    w = copy.deepcopy(w_in)  #避免在函数内部修改全局w
    b = b_in
    w = w.reshape(-1,1)      #为矩阵运算做准备
    y = y.reshape(-1,1)

    for i in range(num_iters):

        # 计算梯度并更新参数
        dj_db,dj_dw = compute_gradient_matrix(X, y, w, b, logistic, lambda_)

        # 使用w、b、alpha和梯度更新参数
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # 保存每次迭代的代价J
        if i<100000:      # 防止资源耗尽
            J_history.append( compute_cost_matrix(X, y, w, b, logistic, lambda_) )

        # 每隔10次或少于10次时打印代价
        if i% math.ceil(num_iters / 10) == 0:
            if verbose: print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w.reshape(w_in.shape), b, J_history  #返回最终的w,b和J历史用于绘图

def zscore_normalize_features(X):
    """
    计算X，按列z-score归一化

    参数:
      X (ndarray): Shape (m,n) 输入数据，m个样本，n个特征

    返回:
      X_norm (ndarray): Shape (m,n)  按列归一化的输入
      mu (ndarray):     Shape (n,)   每个特征的均值
      sigma (ndarray):  Shape (n,)   每个特征的标准差
    """
    # 找到每列/每个特征的均值
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # 找到每列/每个特征的标准差
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # 逐元素地，从每个样本中减去该列的mu，除以该列的std
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma

#检查我们的工作
#from sklearn.preprocessing import scale
#scale(X_orig, axis=0, with_mean=True, with_std=True, copy=True)

######################################################
# 通用绘图例程
######################################################


def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc='best' ):
    """ 绘制带有两个轴的逻辑数据 """
    # 找到正例和负例的索引
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1,)  #work with 1D or 1D y vectors
    neg = neg.reshape(-1,)

    # 绘制样本
    ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c = 'red', label=pos_label)
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, label=neg_label, facecolors='none', edgecolors=dlblue, lw=3)
    ax.legend(loc=loc)

    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False

def plt_tumor_data(x, y, ax):
    """ 在一个轴上绘制肿瘤数据 """
    pos = y == 1
    neg = y == 0

    ax.scatter(x[pos], y[pos], marker='x', s=80, c = 'red', label="恶性")
    ax.scatter(x[neg], y[neg], marker='o', s=100, label="良性", facecolors='none', edgecolors=dlblue,lw=3)
    ax.set_ylim(-0.175,1.1)
    ax.set_ylabel('y')
    ax.set_xlabel('肿瘤大小')
    ax.set_title("分类数据上的逻辑回归")

    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False

# 在0.5处绘制阈值
def draw_vthresh(ax,x):
    """ 绘制阈值 """
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.fill_between([xlim[0], x], [ylim[1], ylim[1]], alpha=0.2, color=dlblue)
    ax.fill_between([x, xlim[1]], [ylim[1], ylim[1]], alpha=0.2, color=dldarkred)
    ax.annotate("z >= 0", xy= [x,0.5], xycoords='data',
                xytext=[30,5],textcoords='offset points')
    d = FancyArrowPatch(
        posA=(x, 0.5), posB=(x+3, 0.5), color=dldarkred,
        arrowstyle='simple, head_width=5, head_length=10, tail_width=0.0',
    )
    ax.add_artist(d)
    ax.annotate("z < 0", xy= [x,0.5], xycoords='data',
                 xytext=[-50,5],textcoords='offset points', ha='left')
    f = FancyArrowPatch(
        posA=(x, 0.5), posB=(x-3, 0.5), color=dlblue,
        arrowstyle='simple, head_width=5, head_length=10, tail_width=0.0',
    )
    ax.add_artist(f)
