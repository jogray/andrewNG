import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

import numpy as np

def test_eval_mse(target):
    y_hat = np.array([2.4, 4.2])
    y_tmp = np.array([2.3, 4.1])
    result = target(y_hat, y_tmp)
    
    assert np.isclose(result, 0.005, atol=1e-6), f"值错误。期望 0.005，得到 {result}"
    
    y_hat = np.array([3.] * 10)
    y_tmp = np.array([3.] * 10)
    result = target(y_hat, y_tmp)
    assert np.isclose(result, 0.), f"值错误。当 y_hat == t_tmp 时期望 0.0，但得到 {result}"
    
    y_hat = np.array([3.])
    y_tmp = np.array([0.])
    result = target(y_hat, y_tmp)
    assert np.isclose(result, 4.5), f"值错误。期望 4.5，但得到 {result}。记住平方项"
    
    y_hat = np.array([3.] * 5)
    y_tmp = np.array([2.] * 5)
    result = target(y_hat, y_tmp)
    assert np.isclose(result, 0.5), f"值错误。期望 0.5，但得到 {result}。记住除以 (2*m)"
    
    print("\033[92m 所有测试通过。")
    
def test_eval_cat_err(target):
    y_hat = np.array([1, 0, 1, 1, 1, 0])
    y_tmp = np.array([0, 1, 0, 0, 0, 1])
    result = target(y_hat, y_tmp)
    assert not np.isclose(result, 6.), f"值错误。期望 1，但得到 {result}。你是否除以了 m？"
    
    y_hat = np.array([1, 2, 0])
    y_tmp = np.array([1, 2, 3])
    result = target(y_hat, y_tmp)
    assert np.isclose(result, 1./3., atol=1e-6), f"值错误。期望 0.333，但得到 {result}"
    
    y_hat = np.array([1, 0, 1, 1, 1, 0])
    y_tmp = np.array([1, 1, 1, 0, 0, 0])
    result = target(y_hat, y_tmp)
    assert np.isclose(result, 3./6., atol=1e-6), f"值错误。期望 0.5，但得到 {result}"
    
    y_hat = np.array([[1], [2], [0], [3]])
    y_tmp = np.array([[1], [2], [1], [3]])
    res_tmp =  target(y_hat, y_tmp)
    assert type(res_tmp) != np.ndarray, f"输出必须是标量，但得到 {type(res_tmp)}"
    
    print("\033[92m 所有测试通过。")
    
def model_test(target, classes, input_size):
    target.build(input_shape=(None,input_size))
    expected_lr = 0.01
    
    assert len(target.layers) == 3, \
        f"层数错误。期望 3，但得到 {len(target.layers)}"
    assert list(target.input_shape) == [None, input_size], \
        f"输入形状错误。期望 [None,  {input_size}]，但得到 {list(target.input_shape)}"
    i = 0
    expected = [[Dense, [None, 120], relu],
                [Dense, [None, 40], relu],
                [Dense, [None, classes], linear]]

    for layer in target.layers:
        assert type(layer) == expected[i][0], \
            f"第 {i} 层类型错误。期望 {expected[i][0]}，但得到 {type(layer)}"
        assert [None, layer.units] == expected[i][1], \
            f"第 {i} 层单元数错误。期望 {expected[i][1]}，但得到 [None, {layer.units}]"
        assert layer.activation == expected[i][2], \
            f"第 {i} 层激活函数错误。期望 {expected[i][2]}，但得到 {layer.activation}"
        assert layer.kernel_regularizer == None, "不得为任何层指定正则化器"
        i = i + 1
        
    assert type(target.loss)==SparseCategoricalCrossentropy, f"损失函数错误。期望 {SparseCategoricalCrossentropy}，但得到 {target.loss}"
    assert type(target.optimizer)==Adam, f"优化器错误。期望 {Adam}，但得到 {target.optimizer}"
    lr = target.optimizer.learning_rate.numpy()
    assert np.isclose(lr, expected_lr, atol=1e-8), f"学习率错误。期望 {expected_lr}，但得到 {lr}"
    assert target.loss.get_config()['from_logits'], f"请在损失函数中设置 from_logits=True"

    print("\033[92m所有测试通过！")
    
def model_s_test(target, classes, input_size):
    target.build(input_shape=(None,input_size))
    expected_lr = 0.01
    
    assert len(target.layers) == 2, \
        f"层数错误。期望 2，但得到 {len(target.layers)}"
    assert list(target.input_shape) == [None, input_size], \
        f"输入形状错误。期望 [None,  {input_size}]，但得到 {list(target.input_shape)}"
    i = 0
    expected = [[Dense, [None, 6], relu],
                [Dense, [None, classes], linear]]

    for layer in target.layers:
        assert type(layer) == expected[i][0], \
            f"第 {i} 层类型错误。期望 {expected[i][0]}，但得到 {type(layer)}"
        assert [None, layer.units] == expected[i][1], \
            f"第 {i} 层单元数错误。期望 {expected[i][1]}，但得到 [None, {layer.units}]"
        assert layer.activation == expected[i][2], \
            f"第 {i} 层激活函数错误。期望 {expected[i][2]}，但得到 {layer.activation}"
        assert layer.kernel_regularizer == None, "不得为任何层指定正则化器"
        i = i + 1
        
    assert type(target.loss)==SparseCategoricalCrossentropy, f"损失函数错误。期望 {SparseCategoricalCrossentropy}，但得到 {target.loss}"
    assert type(target.optimizer)==Adam, f"优化器错误。期望 {Adam}，但得到 {target.optimizer}"
    lr = target.optimizer.learning_rate.numpy()
    assert np.isclose(lr, expected_lr, atol=1e-8), f"学习率错误。期望 {expected_lr}，但得到 {lr}"
    assert target.loss.get_config()['from_logits'], f"请在损失函数中设置 from_logits=True"

    print("\033[92m所有测试通过！")
    
def model_r_test(target, classes, input_size):
    target.build(input_shape=(None,input_size))
    expected_lr = 0.01
    print("ddd")
    assert len(target.layers) == 3, \
        f"层数错误。期望 3，但得到 {len(target.layers)}"
    assert list(target.input_shape) == [None, input_size], \
        f"输入形状错误。期望 [None,  {input_size}]，但得到 {list(target.input_shape)}"
    i = 0
    expected = [[Dense, [None, 120], relu, (tf.keras.regularizers.l2, 0.1)],
                [Dense, [None, 40], relu, (tf.keras.regularizers.l2, 0.1)],
                [Dense, [None, classes], linear, None]]

    for layer in target.layers:
        assert type(layer) == expected[i][0], \
            f"第 {i} 层类型错误。期望 {expected[i][0]}，但得到 {type(layer)}"
        assert [None, layer.units] == expected[i][1], \
            f"第 {i} 层单元数错误。期望 {expected[i][1]}，但得到 [None, {layer.units}]"
        assert layer.activation == expected[i][2], \
            f"第 {i} 层激活函数错误。期望 {expected[i][2]}，但得到 {layer.activation}"
        if not (expected[i][3] == None):
            assert type(layer.kernel_regularizer) == expected[i][3][0], f"正则化器错误。期望 L2 正则化器，但得到 {type(layer.kernel_regularizer)}"
            assert np.isclose(layer.kernel_regularizer.l2,  expected[i][3][1]), f"正则化系数错误。期望 {expected[i][3][1]}，但得到 {layer.kernel_regularizer.l2}"
        else:
            assert layer.kernel_regularizer == None, "不得为第 3 层指定正则化器"
        i = i + 1
        
    assert type(target.loss)==SparseCategoricalCrossentropy, f"损失函数错误。期望 {SparseCategoricalCrossentropy}，但得到 {target.loss}"
    assert type(target.optimizer)==Adam, f"优化器错误。期望 {Adam}，但得到 {target.optimizer}"
    lr = target.optimizer.learning_rate.numpy()
    assert np.isclose(lr, expected_lr, atol=1e-8), f"学习率错误。期望 {expected_lr}，但得到 {lr}"
    assert target.loss.get_config()['from_logits'], f"请在损失函数中设置 from_logits=True"

    print("\033[92m所有测试通过！")
