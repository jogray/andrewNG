"""
plt_quad_logistic.py
    展示逻辑回归的交互式绘图和支持例程
"""

import time
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from matplotlib.patches import FancyArrowPatch
from ipywidgets import Output
from lab_utils_common import np, plt, dlc, dlcolors, sigmoid, compute_cost_matrix, gradient_descent

# for debug
#output = Output() # sends hidden error messages to display when using widgets
#display(output)

class plt_quad_logistic:
    ''' 绘制展示逻辑回归的四合一图 '''
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-locals
    # pylint: disable=missing-function-docstring
    # pylint: disable=attribute-defined-outside-init
    def __init__(self, x_train,y_train, w_range, b_range):
        # setup figure
        fig = plt.figure( figsize=(10,6))
        fig.canvas.toolbar_visible = False
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.set_facecolor('#ffffff') #白色
        gs  = GridSpec(2, 2, figure=fig)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0],  projection='3d')
        ax3 = fig.add_subplot(gs[1,1])
        pos = ax3.get_position().get_points()  ##[[lb_x,lb_y], [rt_x, rt_y]]
        h = 0.05 
        width = 0.2
        axcalc   = plt.axes([pos[1,0]-width, pos[1,1]-h, width, h])  #lx,by,w,h
        ax = np.array([ax0, ax1, ax2, ax3, axcalc])
        self.fig = fig
        self.ax = ax
        self.x_train = x_train
        self.y_train = y_train

        self.w = 0. #initial point, non-array
        self.b = 0.

        # initialize subplots
        self.dplot = data_plot(ax[0], x_train, y_train, self.w, self.b)
        self.con_plot = contour_and_surface_plot(ax[1], ax[2], x_train, y_train, w_range, b_range, self.w, self.b)
        self.cplot = cost_plot(ax[3])

        # setup events
        self.cid = fig.canvas.mpl_connect('button_press_event', self.click_contour)
        self.bcalc = Button(axcalc, '运行梯度下降\n从当前w,b开始（点击）', color=dlc["dlorange"])
        self.bcalc.on_clicked(self.calc_logistic)

#    @output.capture()  # debug
    def click_contour(self, event):
        ''' 在等高线图中点击时调用 '''
        if event.inaxes == self.ax[1]:   #contour plot
            self.w = event.xdata
            self.b = event.ydata

            self.cplot.re_init()
            self.dplot.update(self.w, self.b)
            self.con_plot.update_contour_wb_lines(self.w, self.b)
            self.con_plot.path.re_init(self.w, self.b)

            self.fig.canvas.draw()

#    @output.capture()  # debug
    def calc_logistic(self, event):
        ''' 运行梯度事件时调用 '''
        for it in [1, 8,16,32,64,128,256,512,1024,2048,4096]:
            w, self.b, J_hist = gradient_descent(self.x_train.reshape(-1,1), self.y_train.reshape(-1,1),
                                                 np.array(self.w).reshape(-1,1), self.b, 0.1, it,
                                                 logistic=True, lambda_=0, verbose=False)
            self.w = w[0,0]
            self.dplot.update(self.w, self.b)
            self.con_plot.update_contour_wb_lines(self.w, self.b)
            self.con_plot.path.add_path_item(self.w,self.b)
            self.cplot.add_cost(J_hist)

            time.sleep(0.3)
            self.fig.canvas.draw()


class data_plot:
    ''' 处理数据绘图 '''
    # pylint: disable=missing-function-docstring
    # pylint: disable=attribute-defined-outside-init
    def __init__(self, ax, x_train, y_train, w, b):
        self.ax = ax
        self.x_train = x_train
        self.y_train = y_train
        self.m = x_train.shape[0]
        self.w = w
        self.b = b

        self.plt_tumor_data()
        self.draw_logistic_lines(firsttime=True)
        self.mk_cost_lines(firsttime=True)

        self.ax.autoscale(enable=False) # leave plot scales the same after initial setup

    def plt_tumor_data(self):
        x = self.x_train
        y = self.y_train
        pos = y == 1
        neg = y == 0
        self.ax.scatter(x[pos], y[pos], marker='x', s=80, c = 'red', label="恶性")
        self.ax.scatter(x[neg], y[neg], marker='o', s=100, label="良性", facecolors='none',
                   edgecolors=dlc["dlblue"],lw=3)
        self.ax.set_ylim(-0.175,1.1)
        self.ax.set_ylabel('y')
        self.ax.set_xlabel('肿瘤大小')
        self.ax.set_title("分类数据上的逻辑回归")

    def update(self, w, b):
        self.w = w
        self.b = b
        self.draw_logistic_lines()
        self.mk_cost_lines()

    def draw_logistic_lines(self, firsttime=False):
        if not firsttime:
            self.aline[0].remove()
            self.bline[0].remove()
            self.alegend.remove()

        xlim  = self.ax.get_xlim()
        x_hat = np.linspace(*xlim, 30)
        y_hat = sigmoid(np.dot(x_hat.reshape(-1,1), self.w) + self.b)
        self.aline = self.ax.plot(x_hat, y_hat, color=dlc["dlblue"],
                                     label="y = sigmoid(z)")
        f_wb = np.dot(x_hat.reshape(-1,1), self.w) + self.b
        self.bline = self.ax.plot(x_hat, f_wb, color=dlc["dlorange"], lw=1,
                                     label=f"z = {np.squeeze(self.w):0.2f}x+({self.b:0.2f})")
        self.alegend = self.ax.legend(loc='upper left')

    def mk_cost_lines(self, firsttime=False):
        ''' 绘制垂直代价线'''
        if not firsttime:
            for artist in self.cost_items:
                artist.remove()
        self.cost_items = []
        cstr = f"cost = (1/{self.m})*("
        ctot = 0
        label = '该点的代价'
        addedbreak = False
        for p in zip(self.x_train,self.y_train):
            f_wb_p = sigmoid(self.w*p[0]+self.b)
            c_p = compute_cost_matrix(p[0].reshape(-1,1), p[1],np.array(self.w), self.b, logistic=True, lambda_=0, safe=True)
            c_p_txt = c_p
            a = self.ax.vlines(p[0], p[1],f_wb_p, lw=3, color=dlc["dlpurple"], ls='dotted', label=label)
            label='' #只显示一次
            cxy = [p[0], p[1] + (f_wb_p-p[1])/2]
            b = self.ax.annotate(f'{c_p_txt:0.1f}', xy=cxy, xycoords='data',color=dlc["dlpurple"],
                        xytext=(5, 0), textcoords='offset points')
            cstr += f"{c_p_txt:0.1f} +"
            if len(cstr) > 38 and addedbreak is False:
                cstr += "\n"
                addedbreak = True
            ctot += c_p
            self.cost_items.extend((a,b))
        ctot = ctot/(len(self.x_train))
        cstr = cstr[:-1] + f") = {ctot:0.2f}"
        ## todo.. 搞清楚如何让这个文本框扩展到子图的宽度
        c = self.ax.text(0.05,0.02,cstr, transform=self.ax.transAxes, color=dlc["dlpurple"])
        self.cost_items.append(c)


class contour_and_surface_plot:
    ''' 将等高线图和曲面图组合在一个类中，因为它们有相似的操作 '''
    # pylint: disable=missing-function-docstring
    # pylint: disable=attribute-defined-outside-init
    def __init__(self, axc, axs, x_train, y_train, w_range, b_range, w, b):

        self.x_train = x_train
        self.y_train = y_train
        self.axc = axc
        self.axs = axs

        #设置有用的范围和公共线性空间
        b_space  = np.linspace(*b_range, 100)
        w_space  = np.linspace(*w_range, 100)

        # 获取w,b范围的代价，用于等高线图和3D图
        tmp_b,tmp_w = np.meshgrid(b_space,w_space)
        z = np.zeros_like(tmp_b)
        for i in range(tmp_w.shape[0]):
            for j in range(tmp_w.shape[1]):
                z[i,j] = compute_cost_matrix(x_train.reshape(-1,1), y_train, tmp_w[i,j], tmp_b[i,j],
                                             logistic=True, lambda_=0, safe=True)
                if z[i,j] == 0:
                    z[i,j] = 1e-9

        ### 绘制等高线图 ###
        CS = axc.contour(tmp_w, tmp_b, np.log(z),levels=12, linewidths=2, alpha=0.7,colors=dlcolors)
        axc.set_title('log(Cost(w,b))')
        axc.set_xlabel('w', fontsize=10)
        axc.set_ylabel('b', fontsize=10)
        axc.set_xlim(w_range)
        axc.set_ylim(b_range)
        self.update_contour_wb_lines(w, b, firsttime=True)
        axc.text(0.7,0.05,"点击选择w,b",  bbox=dict(facecolor='white', ec = 'black'), fontsize = 10,
                transform=axc.transAxes, verticalalignment = 'center', horizontalalignment= 'center')

        #代价函数J(w,b)的曲面图
        axs.plot_surface(tmp_w, tmp_b, z,  cmap = cm.jet, alpha=0.3, antialiased=True)
        axs.plot_wireframe(tmp_w, tmp_b, z, color='k', alpha=0.1)
        axs.set_xlabel("$w$")
        axs.set_ylabel("$b$")
        axs.zaxis.set_rotate_label(False)
        axs.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axs.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axs.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axs.set_zlabel("J(w, b)", rotation=90)
        axs.view_init(30, -120)

        axs.autoscale(enable=False)
        axc.autoscale(enable=False)

        self.path = path(self.w,self.b, self.axc)  # initialize an empty path, avoids existance check

    def update_contour_wb_lines(self, w, b, firsttime=False):
        self.w = w
        self.b = b
        cst = compute_cost_matrix(self.x_train.reshape(-1,1), self.y_train, np.array(self.w), self.b,
                                  logistic=True, lambda_=0, safe=True)

        # 移除线条并在等高线图和3D图上重新添加
        if not firsttime:
            for artist in self.dyn_items:
                artist.remove()
        a = self.axc.scatter(self.w, self.b, s=100, color=dlc["dlblue"], zorder= 10, label="当前w,b的代价")
        b = self.axc.hlines(self.b, self.axc.get_xlim()[0], self.w, lw=4, color=dlc["dlpurple"], ls='dotted')
        c = self.axc.vlines(self.w, self.axc.get_ylim()[0] ,self.b, lw=4, color=dlc["dlpurple"], ls='dotted')
        d = self.axc.annotate(f"Cost: {cst:0.2f}", xy= (self.w, self.b), xytext = (4,4), textcoords = 'offset points',
                           bbox=dict(facecolor='white'), size = 10)
        #在3D曲面图中添加点
        e = self.axs.scatter3D(self.w, self.b, cst , marker='X', s=100)

        self.dyn_items = [a,b,c,d,e]


class cost_plot:
    """ 管理plt_quad_logistic的代价绘图 """
    # pylint: disable=missing-function-docstring
    # pylint: disable=attribute-defined-outside-init
    def __init__(self,ax):
        self.ax = ax
        self.ax.set_ylabel("log(代价)")
        self.ax.set_xlabel("迭代次数")
        self.costs = []
        self.cline = self.ax.plot(0,0, color=dlc["dlblue"])

    def re_init(self):
        self.ax.clear()
        self.__init__(self.ax)

    def add_cost(self,J_hist):
        self.costs.extend(J_hist)
        self.cline[0].remove()
        self.cline = self.ax.plot(self.costs)

class path:
    ''' 在等高线图上跟踪梯度下降路径 '''
    # pylint: disable=missing-function-docstring
    # pylint: disable=attribute-defined-outside-init
    def __init__(self, w, b, ax):
        ''' 路径起点的w, b '''
        self.path_items = []
        self.w = w
        self.b = b
        self.ax = ax

    def re_init(self, w, b):
        for artist in self.path_items:
            artist.remove()
        self.path_items = []
        self.w = w
        self.b = b

    def add_path_item(self, w, b):
        a = FancyArrowPatch(
            posA=(self.w, self.b), posB=(w, b), color=dlc["dlblue"],
            arrowstyle='simple, head_width=5, head_length=10, tail_width=0.0',
        )
        self.ax.add_artist(a)
        self.path_items.append(a)
        self.w = w
        self.b = b

#-----------
# 与逻辑梯度下降实验相关
#----------

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """ 截断颜色映射 """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plt_prob(ax, w_out,b_out):
    """ 绘制决策边界，但包含阴影以表示概率 """
    #设置有用的范围和公共线性空间
    x0_space  = np.linspace(0, 4 , 100)
    x1_space  = np.linspace(0, 4 , 100)

    # 获取x0,x1范围的概率
    tmp_x0,tmp_x1 = np.meshgrid(x0_space,x1_space)
    z = np.zeros_like(tmp_x0)
    for i in range(tmp_x0.shape[0]):
        for j in range(tmp_x1.shape[1]):
            z[i,j] = sigmoid(np.dot(w_out, np.array([tmp_x0[i,j],tmp_x1[i,j]])) + b_out)


    cmap = plt.get_cmap('Blues')
    new_cmap = truncate_colormap(cmap, 0.0, 0.5)
    pcm = ax.pcolormesh(tmp_x0, tmp_x1, z,
                   norm=cm.colors.Normalize(vmin=0, vmax=1),
                   cmap=new_cmap, shading='nearest', alpha = 0.9)
    ax.figure.colorbar(pcm, ax=ax)
