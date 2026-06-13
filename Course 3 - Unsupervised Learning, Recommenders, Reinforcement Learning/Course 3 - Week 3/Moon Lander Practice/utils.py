"""
工具模块，包含 Deep Q-Learning - Lunar Lander
Jupyter notebook (C3_W3_A1_Assignment) 的辅助函数，来自 DeepLearning.AI 的
"无监督学习、推荐系统、强化学习" Coursera 课程。
"""

import base64
import random
from itertools import zip_longest

import imageio
import IPython
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 0  # 伪随机数生成器的种子。
MINIBATCH_SIZE = 64  # 小批量大小。
TAU = 1e-3  # 软更新参数。
E_DECAY = 0.995  # ε-贪婪策略的 ε 衰减率。
E_MIN = 0.01  # ε-贪婪策略的最小 ε 值。


random.seed(SEED)


def get_experiences(memory_buffer):
    """
    从经验回放缓冲区中随机采样一批经验元组。

    从给定的 memory_buffer 中随机采样一批经验元组，并将其转换为
    TensorFlow 张量返回。随机样本的大小由小批量大小 (MINIBATCH_SIZE) 决定。
    
    参数:
        memory_buffer (deque):
            包含经验的双端队列。经验以 namedtuple 的形式存储在经验回放缓冲区中:
            namedtuple("Experience", field_names=["state",
            "action", "reward", "next_state", "done"]).

    返回:
        一个元组 (states, actions, rewards, next_states, done_vals)，其中:

            - states 是智能体的起始状态。
            - actions 是智能体从起始状态采取的动作。
            - rewards 是智能体采取动作后获得的奖励。
            - next_states 是智能体采取动作后的新状态。
            - done_vals 是指示回合是否结束的布尔值。

        所有元组元素都是 TensorFlow 张量，其形状由小批量大小和给定的
        Gym 环境决定。对于 Lunar Lander 环境，states 和 next_states 的形状为
        [MINIBATCH_SIZE, 8]，而 actions、rewards 和 done_vals 的形状为
        [MINIBATCH_SIZE]。所有 TensorFlow 张量的元素类型为 tf.float32。
    """

    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    states = tf.convert_to_tensor(
        np.array([e.state for e in experiences if e is not None]), dtype=tf.float32
    )
    actions = tf.convert_to_tensor(
        np.array([e.action for e in experiences if e is not None]), dtype=tf.float32
    )
    rewards = tf.convert_to_tensor(
        np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32
    )
    next_states = tf.convert_to_tensor(
        np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32
    )
    done_vals = tf.convert_to_tensor(
        np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
        dtype=tf.float32,
    )
    return (states, actions, rewards, next_states, done_vals)


def check_update_conditions(t, num_steps_upd, memory_buffer):
    """
    判断是否满足执行学习更新的条件。

    检查当前时间步 t 是否为 num_steps_upd 的倍数，以及 memory_buffer
    是否有足够的经验元组来填满一个小批量（例如，如果小批量大小为 64，
    则经验回放缓冲区应包含超过 64 个经验元组才能执行学习更新）。
    
    参数:
        t (int):
            当前时间步。
        num_steps_upd (int):
            用于确定执行学习更新频率的时间步数。每经过 num_steps_upd 个
            时间步才执行一次学习更新。
        memory_buffer (deque):
            包含经验的双端队列。经验以 namedtuple 的形式存储在经验回放缓冲区中:
            namedtuple("Experience", field_names=["state",
            "action", "reward", "next_state", "done"]).

    返回:
       一个布尔值，如果条件满足则为 True，否则为 False。
    """

    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > MINIBATCH_SIZE:
        return True
    else:
        return False


def get_new_eps(epsilon):
    """
    更新 ε-贪婪策略的 epsilon 值。
    
    使用给定的 ε 衰减率 (E_DECAY) 逐渐将 epsilon 值降低到最小值 (E_MIN)。

    参数:
        epsilon (float):
            当前的 epsilon 值。

    返回:
       一个浮点数，为更新后的 epsilon 值。
    """

    return max(E_MIN, E_DECAY * epsilon)


def get_action(q_values, epsilon=0.0):
    """
    使用 ε-贪婪策略返回一个动作。

    该函数将根据以下规则返回动作：
        - 以 epsilon 的概率，返回一个随机选择的动作。
        - 以 (1 - epsilon) 的概率，返回 q_values 中最大 Q 值对应的
    
    参数:
        q_values (tf.Tensor):
            Q 网络返回的 Q 值。对于 Lunar Lander 环境，该 TensorFlow 张量的
            形状应为 [1, 4]，其元素类型应为 tf.float32。
        epsilon (float):
            当前的 epsilon 值。

    返回:
       一个动作 (numpy.int64)。对于 Lunar Lander 环境，动作用闭区间 [0,3]
       内的整数表示。
    """

    if random.random() > epsilon:
        return np.argmax(q_values.numpy()[0])
    else:
        return random.choice(np.arange(4))


def update_target_network(q_network, target_q_network):
    """
    使用软更新方式更新目标 Q 网络的权重。
    
    目标 Q 网络 (target_q_network) 的权重使用软更新规则进行更新：
    
                    w_target = (TAU * w) + (1 - TAU) * w_target
    
    其中 w_target 是目标 Q 网络的权重，TAU 是软更新参数，w 是 Q 网络的权重。
    
    参数:
        q_network (tf.keras.Sequential): 
            Q 网络。
        target_q_network (tf.keras.Sequential):
            目标 Q 网络。
    """

    for target_weights, q_net_weights in zip(
        target_q_network.weights, q_network.weights
    ):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)


def plot_history(point_history, **kwargs):
    """
    绘制智能体在每个回合后获得的总分数以及移动平均值（滚动均值）。

    参数:
        point_history (list):
            一个列表，包含智能体在每个回合后获得的总分数。
        **kwargs: 可选参数
            window_size (int):
                用于计算移动平均值（滚动均值）的窗口大小。该整数决定每个窗口
                使用的固定数据点数量。默认窗口大小设置为 point_history 中总数据
                点数的 10%，例如如果 point_history 有 200 个数据点，默认窗口
                大小将为 20。
            lower_limit (int):
                x 轴在数据坐标中的下限。默认值为 0。
            upper_limit (int):
                x 轴在数据坐标中的上限。默认值为 len(point_history)。
            plot_rolling_mean_only (bool):
                如果为 True，则只绘制移动平均值（滚动均值）而不绘制原始数据点。
                默认值为 False。
            plot_data_only (bool):
                如果为 True，则只绘制原始数据点而不绘制移动平均值。
                默认值为 False。
    """

    lower_limit = 0
    upper_limit = len(point_history)

    window_size = (upper_limit * 10) // 100

    plot_rolling_mean_only = False
    plot_data_only = False

    if kwargs:
        if "window_size" in kwargs:
            window_size = kwargs["window_size"]

        if "lower_limit" in kwargs:
            lower_limit = kwargs["lower_limit"]

        if "upper_limit" in kwargs:
            upper_limit = kwargs["upper_limit"]

        if "plot_rolling_mean_only" in kwargs:
            plot_rolling_mean_only = kwargs["plot_rolling_mean_only"]

        if "plot_data_only" in kwargs:
            plot_data_only = kwargs["plot_data_only"]

    points = point_history[lower_limit:upper_limit]

    # 生成用于绘图的 x 轴。
    episode_num = [x for x in range(lower_limit, upper_limit)]

    # 使用 Pandas 计算滚动均值（移动平均）。
    rolling_mean = pd.DataFrame(points).rolling(window_size).mean()

    plt.figure(figsize=(10, 7), facecolor="white")

    if plot_data_only:
        plt.plot(episode_num, points, linewidth=1, color="cyan")
    elif plot_rolling_mean_only:
        plt.plot(episode_num, rolling_mean, linewidth=2, color="magenta")
    else:
        plt.plot(episode_num, points, linewidth=1, color="cyan")
        plt.plot(episode_num, rolling_mean, linewidth=2, color="magenta")

    text_color = "black"

    ax = plt.gca()
    ax.set_facecolor("black")
    plt.grid()
    plt.xlabel("Episode", color=text_color, fontsize=30)
    plt.ylabel("Total Points", color=text_color, fontsize=30)
    yNumFmt = mticker.StrMethodFormatter("{x:,}")
    ax.yaxis.set_major_formatter(yNumFmt)
    ax.tick_params(axis="x", colors=text_color)
    ax.tick_params(axis="y", colors=text_color)
    plt.show()


def display_table(current_state, action, next_state, reward, done):
    """
    显示一个表格，包含 Gym Lunar Lander 环境中的当前状态、动作、下一状态、
    奖励和终止标志值。

    表格中所有浮点数显示时四舍五入到小数点后 3 位，动作使用其标签名称
    而非数值显示（即如果 action = 0，动作将显示为 "Do nothing" 而非 "0"）。

    参数:
        current_state (numpy.ndarray):
            Lunar Lander 环境在采取动作前返回的当前状态向量。
        action (int):
            智能体采取的动作。在 Lunar Lander 环境中，动作用闭区间 [0,3] 内的
            整数表示，对应于：
                - Do nothing = 0
                - Fire right engine = 1
                - Fire main engine = 2
                - Fire left engine = 3
        next_state (numpy.ndarray):
            智能体采取动作后 Lunar Lander 环境返回的状态向量，即使用
            env.step(action) 运行单个时间步后返回的观测值。
        reward (numpy.float64):
            智能体采取动作后 Lunar Lander 环境返回的奖励，即使用
            env.step(action) 运行单个时间步后返回的奖励。
        done (bool):
            智能体采取动作后 Lunar Lander 环境返回的终止标志值，即使用
            env.step(action) 运行单个时间步后返回的终止标志值。
    
    返回:
        table (Pandas Dataframe):
            一个包含 current_state、action、next_state、reward 和 done 值的
            数据框。这将使表格在 Jupyter Notebook 中显示。
    """
    
    STATE_VECTOR_COL_NAME = '状态向量'
    DERIVED_COL_NAME = '从状态向量推导（越接近零越好）'
    
    # 状态
    add_derived_info = lambda state: np.hstack([
        state, 
        [(state[0]**2 + state[1]**2)**.5],
        [(state[2]**2 + state[3]**2)**.5],
        [np.abs(state[4])]
    ])
    
    modified_current_state = add_derived_info(current_state)
    modified_next_state = add_derived_info(next_state)
    
    states = np.vstack([
        modified_current_state, 
        modified_next_state,
        modified_next_state - modified_current_state,        
    ]).T
    
    get_state = lambda idx, type=np.float32: dict(zip(
        ['当前状态', '下一状态'], 
        states[idx].astype(type)
    ))

    # 动作
    action_labels = [
        "不操作",
        "点燃右引擎",
        "点燃主引擎",
        "点燃左引擎",
    ]

    display(
        pd.DataFrame({
            ('', '', ''): {'动作': action_labels[action], '奖励': reward, '回合终止': done},
            (STATE_VECTOR_COL_NAME, '坐标', 'X (水平)'): get_state(0),
            (STATE_VECTOR_COL_NAME, '坐标', 'Y (垂直)'): get_state(1),
            (STATE_VECTOR_COL_NAME, '速度', 'X (水平)'): get_state(2),
            (STATE_VECTOR_COL_NAME, '速度', 'Y (垂直)'): get_state(3),
            (STATE_VECTOR_COL_NAME, '倾斜', '角度'): get_state(4),
            (STATE_VECTOR_COL_NAME, '倾斜', '角速度'): get_state(5),
            (STATE_VECTOR_COL_NAME, '地面接触', '左腿?'): get_state(6, np.bool),
            (STATE_VECTOR_COL_NAME, '地面接触', '右腿?'): get_state(7, np.bool),
            (DERIVED_COL_NAME, '距着陆台距离', ''): get_state(8),
            (DERIVED_COL_NAME, '速度', ''): get_state(9),
            (DERIVED_COL_NAME, '倾斜角度 (绝对值)', ''): get_state(10),
        })\
            .fillna('')\
            .reindex(['当前状态', '动作', '下一状态', '奖励', '回合终止'])\
            .style\
            .applymap(lambda x: 'background-color : grey' if x == '' else '')\
            .set_table_styles(
                [
                    {"selector": "th", "props": [("border", "1px solid grey"), ('text-align', 'center')]},
                    {"selector": "tbody td", "props": [("border", "1px solid grey"), ('text-align', 'center')]},
                ]
            )
    )


def embed_mp4(filename):
    """
    在 Jupyter notebook 中嵌入 MP4 视频文件。
    
    参数:
        filename (string):
            要嵌入的 MP4 视频文件的路径（例如 "./videos/lunar_lander.mp4"）。
    
    返回:
        返回一个来自给定视频文件的 display 对象。这将使视频在 Jupyter Notebook 中显示。
    """

    video = open(filename, "rb").read()
    b64 = base64.b64encode(video)
    tag = """
    <video width="840" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>""".format(
        b64.decode()
    )

    return IPython.display.HTML(tag)


def create_video(filename, env, q_network, fps=30):
    """
    创建智能体与 Gym 环境交互的视频。

    智能体将使用 q_network 将状态映射到 Q 值，并使用贪婪策略选择动作
    （即选择产生最大 Q 值的动作）与给定的 env 环境进行交互。
    
    视频将保存到具有给定文件名的文件中。视频格式必须通过提供文件扩展名
    在文件名中指定（.mp4、.gif 等）。如果您想使用 embed_mp4 函数将视频
    嵌入 Jupyter notebook，则视频必须保存为 MP4 文件。
    
    参数:
        filename (string):
            视频将保存到的文件路径。视频格式将根据文件名自动选择。因此，必须
            通过提供文件扩展名在文件名中指定视频格式（例如
            "./videos/lunar_lander.mp4"）。要查看支持的格式列表，请参阅
            imageio 文档: https://imageio.readthedocs.io/en/v2.8.0/formats.html
        env (Gym Environment): 
            智能体将与之交互的 Gym 环境。
        q_network (tf.keras.Sequential):
            一个将状态映射到 Q 值的 TensorFlow Keras Sequential 模型。
        fps (int):
            每秒帧数。指定输出视频的帧率。默认帧率为每秒 30 帧。
    """

    with imageio.get_writer(filename, fps=fps) as video:
        done = False
        state = env.reset()
        frame = env.render(mode="rgb_array")
        video.append_data(frame)
        while not done:
            state = np.expand_dims(state, axis=0)
            q_values = q_network(state)
            action = np.argmax(q_values.numpy()[0])
            state, _, done, _ = env.step(action)
            frame = env.render(mode="rgb_array")
            video.append_data(frame)
