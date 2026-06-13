"""
lab_utils_common
   包含本周所有实验使用的通用例程和变量定义。
   相比之下，特定的大型绘图例程将放在单独的文件中，
   通常在使用的周次中导入。
   这些文件将导入本文件
"""
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from ipywidgets import Output
from matplotlib.widgets import Button, CheckButtons

np.set_printoptions(precision=2)

dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0', dldarkblue =  '#0D5BDC')
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'; dldarkblue =  '#0D5BDC'
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]
plt.style.use('./deeplearning.mplstyle')

def sigmoid(z):
    """
    计算z的sigmoid值

    参数
    ----------
    z : array_like
        标量或任意大小的numpy数组。

    返回值
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
    使用逻辑损失计算成本，非矩阵版本

    参数:
      X (ndarray): 形状 (m,n)  包含n个特征的样本矩阵
      y (ndarray): 形状 (m,)   目标值
      w (ndarray): 形状 (n,)   预测参数
      b (scalar):               预测参数
      lambda_ : (scalar, float) 控制正则化程度, 0 = 无正则化
      safe : (boolean)          True选择下/溢出安全算法
    返回值:
      cost (scalar): 成本
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
    x   : (ndarray 形状 (n,1) 或 (n,)  输入
    out : (ndarray 形状与x匹配        输出 ~= np.log(1+exp(x))
    '''

    out  = np.zeros_like(x,dtype=float)
    i    = x <= maximum
    ni   = np.logical_not(i)

    out[i]  = np.log(1 + np.exp(x[i]))
    out[ni] = x[ni]
    return out


def compute_cost_matrix(X, y, w, b, logistic=False, lambda_=0, safe=True):
    """
    使用矩阵计算成本
    参数:
      X : (ndarray, 形状 (m,n))          样本矩阵
      y : (ndarray  形状 (m,) 或 (m,1))  每个样本的目标值
      w : (ndarray  形状 (n,) 或 (n,1))  模型参数值
      b : (scalar )                       模型参数值
      verbose : (Boolean) 如果为true, 打印中间值 f_wb
    返回值:
      total_cost: (scalar)                成本
    """
    m = X.shape[0]
    y = y.reshape(-1,1)             # 确保2维
    w = w.reshape(-1,1)             # 确保2维
    if logistic:
        if safe:  #安全防溢出
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
      X : (ndarray, 形状 (m,n))          样本矩阵
      y : (ndarray  形状 (m,) 或 (m,1))  每个样本的目标值
      w : (ndarray  形状 (n,) 或 (n,1))  模型参数值
      b : (scalar )                       模型参数值
      logistic: (boolean)                 如果为false则线性, 如果为true则逻辑回归
      lambda_:  (float)                   非零时应用正则化
    返回值
      dj_dw: (array_like 形状 (n,1))     成本关于参数w的梯度
      dj_db: (scalar)                     成本关于参数b的梯度
    """
    m = X.shape[0]
    y = y.reshape(-1,1)             # 确保2维
    w = w.reshape(-1,1)             # 确保2维

    f_wb  = sigmoid( X @ w + b ) if logistic else  X @ w + b      # (m,n)(n,1) = (m,1)
    err   = f_wb - y                                              # (m,1)
    dj_dw = (1/m) * (X.T @ err)                                   # (n,m)(m,1) = (n,1)
    dj_db = (1/m) * np.sum(err)                                   # scalar

    dj_dw += (lambda_/m) * w        # 正则化                      # (n,1)

    return dj_db, dj_dw                                           # scalar, (n,1)

def gradient_descent(X, y, w_in, b_in, alpha, num_iters, logistic=False, lambda_=0, verbose=True, Trace=True):
    """
    执行批量梯度下降来学习theta。通过以学习率alpha
    执行num_iters步梯度来更新theta

    参数:
      X (ndarray):    形状 (m,n)         样本矩阵
      y (ndarray):    形状 (m,) 或 (m,1) 每个样本的目标值
      w_in (ndarray): 形状 (n,) 或 (n,1) 模型参数的初始值
      b_in (scalar):                      模型参数的初始值
      logistic: (boolean)                 如果为false则线性, 如果为true则逻辑回归
      lambda_:  (float)                   非零时应用正则化
      alpha (float):                      学习率
      num_iters (int):                    运行梯度下降的迭代次数

    返回值:
      w (ndarray): 形状 (n,) 或 (n,1)    更新后的参数值; 匹配输入形状
      b (scalar):                         更新后的参数值
    """
    # 存储每次迭代的成本J和w的数组，主要用于后续绘图
    J_history = []
    w = copy.deepcopy(w_in)  #避免在函数内修改全局w
    b = b_in
    w = w.reshape(-1,1)      #为矩阵运算做准备
    y = y.reshape(-1,1)
    last_cost = np.Inf

    for i in range(num_iters):

        # 计算梯度并更新参数
        dj_db,dj_dw = compute_gradient_matrix(X, y, w, b, logistic, lambda_)

        # 使用w, b, alpha和梯度更新参数
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # 每次迭代保存成本J
        ccost = compute_cost_matrix(X, y, w, b, logistic, lambda_)
        if Trace and i<100000:      # 防止资源耗尽
            J_history.append( ccost )

        # 每隔10次迭代打印一次成本，如果迭代次数少于10则全部打印
        if i% math.ceil(num_iters / 10) == 0:
            if verbose: print(f"迭代 {i:4d}: 成本 {ccost}   ")
            if verbose ==2: print(f"dj_db, dj_dw = {dj_db: 0.3f}, {dj_dw.reshape(-1)}")

            if ccost == last_cost:
                alpha = alpha/10
                print(f" alpha 现在为 {alpha}")
            last_cost = ccost

    return w.reshape(w_in.shape), b, J_history  #返回最终的w, b和J历史用于绘图

def zscore_normalize_features(X):
    """
    计算X, 按列zscore标准化

    参数:
      X (ndarray): 形状 (m,n) 输入数据, m个样本, n个特征

    返回值:
      X_norm (ndarray): 形状 (m,n)  按列标准化的输入
      mu (ndarray):     形状 (n,)   每个特征的均值
      sigma (ndarray):  形状 (n,)   每个特征的标准差
    """
    # 求每列/特征的均值
    mu     = np.mean(X, axis=0)                 # mu的形状为 (n,)
    # 求每列/特征的标准差
    sigma  = np.std(X, axis=0)                  # sigma的形状为 (n,)
    # 逐元素地，从每个样本中减去该列的mu，除以该列的标准差
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma

#检查我们的工作
#from sklearn.preprocessing import scale
#scale(X_orig, axis=0, with_mean=True, with_std=True, copy=True)

######################################################
# 通用绘图例程
######################################################


def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc='best' ):
    """ 绘制包含两个轴的逻辑回归数据 """
    # 找到正例和负例的索引
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1,)  #处理1D或1D y向量
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


#-----------------------------------------------------
# 通用交互式绘图例程
#-----------------------------------------------------

class button_manager:
    ''' 处理matplotlib检查按钮的一些缺失功能
    初始化时:
        创建按钮, 链接到button_click例程,
        使用活动索引和firsttime=True调用call_on_click
    点击时:
        维持单个按钮开启状态, 调用call_on_click
    '''

    #@output.capture()  # debug
    def __init__(self,fig, dim, labels, init, call_on_click):
        '''
        dim: (list)     [leftbottom_x,bottom_y,width,height]
        labels: (list)  例如 ['1','2','3','4','5','6']
        init: (list)    例如 [True, False, False, False, False, False]
        '''
        self.fig = fig
        self.ax = plt.axes(dim)  #lx,by,w,h
        self.init_state = init
        self.call_on_click = call_on_click
        self.button  = CheckButtons(self.ax,labels,init)
        self.button.on_clicked(self.button_click)
        self.status = self.button.get_status()
        self.call_on_click(self.status.index(True),firsttime=True)

    #@output.capture()  # debug
    def reinit(self):
        self.status = self.init_state
        self.button.set_active(self.status.index(True))      #关闭旧的, 将触发更新并设置为status

    #@output.capture()  # debug
    def button_click(self, event):
        ''' 维持单个按钮开启状态。如果点击了开启按钮, 将正确处理 '''
        #new_status = self.button.get_status()
        #new = [self.status[i] ^ new_status[i] for i in range(len(self.status))]
        #newidx = new.index(True)
        self.button.eventson = False
        self.button.set_active(self.status.index(True))  #关闭旧的或重新启用相同的
        self.button.eventson = True
        self.status = self.button.get_status()
        self.call_on_click(self.status.index(True))
