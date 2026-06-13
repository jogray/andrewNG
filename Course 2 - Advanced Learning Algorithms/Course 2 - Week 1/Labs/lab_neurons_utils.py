import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
from matplotlib import cm
import matplotlib.colors as colors
from lab_utils_common import dlc

def plt_prob_1d(ax,fwb):
    """ 绘制决策边界，包含阴影以表示概率 """
    #设置有用的范围和常用线性空间
    x_space  = np.linspace(0, 5 , 50)
    y_space  = np.linspace(0, 1 , 50)

    # 获取x范围的概率，扩展到y
    z = np.zeros((len(x_space),len(y_space)))
    for i in range(len(x_space)):
        x = np.array([[x_space[i]]])
        z[:,i] = fwb(x)

    cmap = plt.get_cmap('Blues')
    new_cmap = truncate_colormap(cmap, 0.0, 0.5)
    pcm = ax.pcolormesh(x_space, y_space, z,
                   norm=cm.colors.Normalize(vmin=0, vmax=1),
                   cmap=new_cmap, shading='nearest', alpha = 0.9)
    ax.figure.colorbar(pcm, ax=ax)
    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """ 截断颜色映射 """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def sigmoidnp(z):
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

def plt_linear(X_train, Y_train, prediction_tf, prediction_np):
    fig, ax = plt.subplots(1,2, figsize=(16,4))
    ax[0].scatter(X_train, Y_train, marker='x', c='r', label="数据点")
    ax[0].plot(X_train, prediction_tf,  c=dlc['dlblue'], label="模型输出")
    ax[0].text(1.6,350,r"y=$200 x + 100$", fontsize='xx-large', color=dlc['dlmagenta'])
    ax[0].legend(fontsize='xx-large')
    ax[0].set_ylabel('价格（千美元）', fontsize='xx-large')
    ax[0].set_xlabel('面积（千平方英尺）', fontsize='xx-large')
    ax[0].set_title("Tensorflow预测",fontsize='xx-large')

    ax[1].scatter(X_train, Y_train, marker='x', c='r', label="数据点")
    ax[1].plot(X_train, prediction_np,  c=dlc['dlblue'], label="模型输出")
    ax[1].text(1.6,350,r"y=$200 x + 100$", fontsize='xx-large', color=dlc['dlmagenta'])
    ax[1].legend(fontsize='xx-large')
    ax[1].set_ylabel('价格（千美元）', fontsize='xx-large')
    ax[1].set_xlabel('面积（千平方英尺）', fontsize='xx-large')
    ax[1].set_title("Numpy预测",fontsize='xx-large')
    plt.show()
    
    
def plt_logistic(X_train, Y_train, model, set_w, set_b, pos, neg):
    fig,ax = plt.subplots(1,2,figsize=(16,4))

    layerf= lambda x : model.predict(x)
    plt_prob_1d(ax[0], layerf)

    ax[0].scatter(X_train[pos], Y_train[pos], marker='x', s=80, c = 'red', label="y=1")
    ax[0].scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0", facecolors='none', 
                  edgecolors=dlc["dlblue"],lw=3)

    ax[0].set_ylim(-0.08,1.1)
    ax[0].set_xlim(-0.5,5.5)
    ax[0].set_ylabel('y', fontsize=16)
    ax[0].set_xlabel('x', fontsize=16)
    ax[0].set_title('Tensorflow模型', fontsize=20)
    ax[0].legend(fontsize=16)

    layerf= lambda x : sigmoidnp(np.dot(set_w,x.reshape(1,1)) + set_b)
    plt_prob_1d(ax[1], layerf)

    ax[1].scatter(X_train[pos], Y_train[pos], marker='x', s=80, c = 'red', label="y=1")
    ax[1].scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0", facecolors='none', 
                  edgecolors=dlc["dlblue"],lw=3)

    ax[1].set_ylim(-0.08,1.1)
    ax[1].set_xlim(-0.5,5.5)
    ax[1].set_ylabel('y', fontsize=16)
    ax[1].set_xlabel('x', fontsize=16)
    ax[1].set_title('Numpy模型', fontsize=20)
    ax[1].legend(fontsize=16)
    plt.show()
