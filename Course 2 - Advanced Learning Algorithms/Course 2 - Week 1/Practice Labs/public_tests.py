# 单元测试
from tensorflow.keras.activations import sigmoid as tf_keras_sigmoid
from tensorflow.keras.layers import Dense

import numpy as np

def test_c1(target):
    assert len(target.layers) == 3, \
        f"层数错误。期望3但得到 {len(target.layers)}"
    # 兼容 Keras 2 (target.input.shape.as_list()) 和 Keras 3 (target.input_shape)
    try:
        input_shape = target.input.shape.as_list()
    except AttributeError:
        input_shape = list(target.input_shape)
    assert input_shape == [None, 400], \
        f"输入形状错误。期望 [None,  400] 但得到 {input_shape}"
    i = 0
    expected = [[Dense, [None, 25], tf_keras_sigmoid],
                [Dense, [None, 15], tf_keras_sigmoid],
                [Dense, [None, 1], tf_keras_sigmoid]]

    for layer in target.layers:
        assert type(layer) == expected[i][0], \
            f"第{i}层类型错误。期望 {expected[i][0]} 但得到 {type(layer)}"
        # 兼容 Keras 2 (.shape.as_list()) 和 Keras 3 (.shape 是 tuple)
        try:
            output_shape = layer.output.shape.as_list()
        except AttributeError:
            output_shape = list(layer.output.shape)
        assert output_shape == expected[i][1], \
            f"第{i}层单元数错误。期望 {expected[i][1]} 但得到 {output_shape}"
        assert layer.activation == expected[i][2], \
            f"第{i}层激活函数错误。期望 {expected[i][2]} 但得到 {layer.activation}"
        i = i + 1

    print("\033[92m所有测试通过!")
    
def test_c2(target):
    
    def linear(a):
        return a
    
    def linear_times3(a):
        return a * 3
    
    x_tst = np.array([1., 2., 3., 4.])  # (1 examples, 3 features)
    W_tst = np.array([[1., 2.], [1., 2.], [1., 2.], [1., 2.]]) # (3 input features, 2 output features)
    b_tst = np.array([0., 0.])  # (2 features)
    
    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert A_tst.shape[0] == len(b_tst)
    assert np.allclose(A_tst, [10., 20.]), \
        "输出错误。检查点积"
    
    b_tst = np.array([3., 5.])  # (2 features)
    
    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert np.allclose(A_tst, [13., 25.]), \
        "输出错误。检查公式中的偏置项"
    
    A_tst = target(x_tst, W_tst, b_tst, linear_times3)
    assert np.allclose(A_tst, [39., 75.]), \
        "输出错误。你是否在最后应用了激活函数？"
    
    print("\033[92m所有测试通过!")
    
def test_c3(target):
    
    def linear(a):
        return a
    
    def linear_times3(a):
        return a * 3
    
    x_tst = np.array([1., 2., 3., 4.])  # (1 examples, 3 features)
    W_tst = np.array([[1., 2.], [1., 2.], [1., 2.], [1., 2.]]) # (3 input features, 2 output features)
    b_tst = np.array([0., 0.])  # (2 features)
    
    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert A_tst.shape[0] == len(b_tst)
    assert np.allclose(A_tst, [10., 20.]), \
        "输出错误。检查点积"
    
    b_tst = np.array([3., 5.])  # (2 features)
    
    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert np.allclose(A_tst, [13., 25.]), \
        "输出错误。检查公式中的偏置项"
    
    A_tst = target(x_tst, W_tst, b_tst, linear_times3)
    assert np.allclose(A_tst, [39., 75.]), \
        "输出错误。你是否在最后应用了激活函数？"
    
    x_tst = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])  # (2 examples, 4 features)
    W_tst = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12]]) # (3 input features, 2 output features)
    b_tst = np.array([0., 0., 0.])  # (2 features)
    
    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert A_tst.shape == (2, 3)
    assert np.allclose(A_tst, [[ 70.,  80.,  90.], [158., 184., 210.]]), \
        "输出错误。检查点积"
    
    b_tst = np.array([3., 5., 6])  # (3 features)
    
    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert np.allclose(A_tst, [[ 73.,  85.,  96.], [161., 189., 216.]]), \
        "输出错误。检查公式中的偏置项"
    
    A_tst = target(x_tst, W_tst, b_tst, linear_times3)
    assert np.allclose(A_tst, [[ 219.,  255.,  288.], [483., 567., 648.]]), \
        "输出错误。你是否在最后应用了激活函数？"
    
    print("\033[92m所有测试通过!")  
