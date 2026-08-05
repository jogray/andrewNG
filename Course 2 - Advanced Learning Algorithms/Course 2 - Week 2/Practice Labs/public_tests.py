import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, sigmoid, relu

def test_my_softmax(target):
    z = np.array([1., 2., 3., 4.])
    a = target(z)
    atf = tf.nn.softmax(z)
    
    assert np.allclose(a, atf, atol=1e-10),         f"错误的值。期望 {atf}, 得到 {a}"
    
    z = np.array([np.log(0.1)] * 10)
    a = target(z)
    atf = tf.nn.softmax(z)
    
    assert np.allclose(a, atf, atol=1e-10),         f"错误的值。期望 {atf}, 得到 {a}"
    
    print("\033[92m 所有测试通过.")
    
def test_model(target, classes, input_size):
    target.build(input_shape=(None,input_size))
    # Call the model with dummy data to initialize layers for Keras 3 compatibility
    target(np.zeros((1, input_size)))
    
    assert len(target.layers) == 3, \
        f"层数错误。期望3但得到 {len(target.layers)}"
    assert list(target.input_shape) == [None, input_size], \
        f"输入形状错误。期望 [None,  {input_size}] 但得到 {list(target.input_shape)}"
    i = 0
    expected = [[Dense, [None, 25], relu],
                [Dense, [None, 15], relu],
                [Dense, [None, classes], linear]]

    for layer in target.layers:
        assert type(layer) == expected[i][0], \
            f"第{i}层类型错误。期望 {expected[i][0]} 但得到 {type(layer)}"
        assert list(layer.output.shape) == expected[i][1], \
            f"第{i}层单元数错误。期望 {expected[i][1]} 但得到 {list(layer.output.shape)}"
        assert layer.activation == expected[i][2], \
            f"第{i}层激活函数错误。期望 {expected[i][2]} 但得到 {layer.activation}"
        i = i + 1

    print("\033[92m所有测试通过!")
    