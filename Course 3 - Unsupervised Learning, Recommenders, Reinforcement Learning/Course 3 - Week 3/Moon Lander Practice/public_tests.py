from tensorflow.keras.activations import relu, linear
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import numpy as np

def test_network(target):
    num_actions = 4
    state_size = 8
    i = 0
    assert len(target.layers) == 3, f"层数错误。期望 3 但得到 {len(target.layers)}"
    assert target.input.shape.as_list() == [None, state_size], \
        f"输入形状错误。期望 [None, 400] 但得到 {target.input.shape.as_list()}" 
    expected = [[Dense, [None, 64], relu],
                [Dense, [None, 64], relu],
                [Dense, [None, num_actions], linear]]

    for layer in target.layers:
        assert type(layer) == expected[i][0], \
            f"第 {i} 层类型错误。期望 {expected[i][0]} 但得到 {type(layer)}"
        assert layer.output.shape.as_list() == expected[i][1], \
            f"第 {i} 层单元数错误。期望 {expected[i][1]} 但得到 {layer.output.shape.as_list()}"
        assert layer.activation == expected[i][2], \
            f"第 {i} 层激活函数错误。期望 {expected[i][2]} 但得到 {layer.activation}"
        i = i + 1

    print("\033[92m所有测试通过!")
    
def test_optimizer(target, ALPHA):
    assert type(target) == Adam, f"优化器错误。期望: {Adam}, 得到: {target}"
    assert np.isclose(target.learning_rate.numpy(), ALPHA), f"学习率错误。期望: {ALPHA}, 得到: {target.learning_rate.numpy()}"
    print("\033[92m所有测试通过!")
    
    
def test_compute_loss(target):
    num_actions = 4
    def target_q_network_random(inputs):
        return np.float32(np.random.rand(inputs.shape[0],num_actions))
    
    def q_network_random(inputs):
        return np.float32(np.random.rand(inputs.shape[0],num_actions))
    
    def target_q_network_ones(inputs):
        return np.float32(np.ones((inputs.shape[0], num_actions)))
    
    def q_network_ones(inputs):
        return np.float32(np.ones((inputs.shape[0], num_actions)))
    
    np.random.seed(1)
    states = np.float32(np.random.rand(64, 8))
    actions = np.float32(np.floor(np.random.uniform(0, 1, (64, )) * 4))
    rewards = np.float32(np.random.rand(64, ))
    next_states = np.float32(np.random.rand(64, 8))
    done_vals = np.float32((np.random.uniform(0, 1, size=(64,)) > 0.96) * 1)

    loss = target((states, actions, rewards, next_states, done_vals), 0.995, q_network_random, target_q_network_random)
    

    assert np.isclose(loss, 0.6991737), f"值错误。期望 {0.6991737}, 得到 {loss}"

    # 测试回合终止时的情况
    done_vals = np.float32(np.ones((64,)))
    loss = target((states, actions, rewards, next_states, done_vals), 0.995, q_network_ones, target_q_network_ones)
    assert np.isclose(loss, 0.343270182), f"值错误。期望 {0.343270182}, 得到 {loss}"
      
    # 测试参数 A = B 时的 MSE
    done_vals = np.float32((np.random.uniform(0, 1, size=(64,)) > 0.96) * 1)
    rewards = np.float32(np.ones((64, )))
    loss = target((states, actions, rewards, next_states, done_vals), 0, q_network_ones, target_q_network_ones)
    assert np.isclose(loss, 0), f"值错误。期望 {0}, 得到 {loss}"
 
    # 测试参数 A = 0 且 B = 1 时的 MSE
    done_vals = np.float32((np.random.uniform(0, 1, size=(64,)) > 0.96) * 1)
    rewards = np.float32(np.zeros((64, )))
    loss = target((states, actions, rewards, next_states, done_vals), 0, q_network_ones, target_q_network_ones)
    assert np.isclose(loss, 1), f"值错误。期望 {1}, 得到 {loss}"

    print("\033[92m所有测试通过!")
    