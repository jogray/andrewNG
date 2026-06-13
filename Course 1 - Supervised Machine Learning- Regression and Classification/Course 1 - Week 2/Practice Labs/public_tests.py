import numpy as np

def compute_cost_test(target):
    # print("Using X with shape (4, 1)")
    # 情况 1
    x = np.array([2, 4, 6, 8]).T
    y = np.array([7, 11, 15, 19]).T
    initial_w = 2
    initial_b = 3.0
    cost = target(x, y, initial_w, initial_b)
    assert cost == 0, f"情况1: 完美预测时代价必须为0，但得到 {cost}"
    
    # 情况 2
    x = np.array([2, 4, 6, 8]).T
    y = np.array([7, 11, 15, 19]).T
    initial_w = 2.0
    initial_b = 1.0
    cost = target(x, y, initial_w, initial_b)
    assert cost == 2, f"情况2: 代价必须为2，但得到 {cost}"
    
    # print("Using X with shape (5, 1)")
    # 情况 3
    x = np.array([1.5, 2.5, 3.5, 4.5, 1.5]).T
    y = np.array([4, 7, 10, 13, 5]).T
    initial_w = 1
    initial_b = 0.0
    cost = target(x, y, initial_w, initial_b)
    assert np.isclose(cost, 15.325), f"情况3: 完美预测时代价必须为15.325，但得到 {cost}"
    
    # 情况 4
    initial_b = 1.0
    cost = target(x, y, initial_w, initial_b)
    assert np.isclose(cost, 10.725), f"情况4: 代价必须为10.725，但得到 {cost}"
    
    # 情况 5
    y = y - 2
    initial_b = 1.0
    cost = target(x, y, initial_w, initial_b)
    assert  np.isclose(cost, 4.525), f"情况5: 代价必须为4.525，但得到 {cost}"
    
    print("\033[92m所有测试通过!")
    
def compute_gradient_test(target):
    print("使用形状为 (4, 1) 的X")
    # 情况 1
    x = np.array([2, 4, 6, 8]).T
    y = np.array([4.5, 8.5, 12.5, 16.5]).T
    initial_w = 2.
    initial_b = 0.5
    dj_dw, dj_db = target(x, y, initial_w, initial_b)
    #assert dj_dw.shape == initial_w.shape, f"Wrong shape for dj_dw. {dj_dw} != {initial_w.shape}"
    assert dj_db == 0.0, f"情况1: dj_db错误: {dj_db} != 0.0"
    assert np.allclose(dj_dw, 0), f"情况1: dj_dw错误: {dj_dw} != [[0.0]]"
    
    # 情况 2 
    x = np.array([2, 4, 6, 8]).T
    y = np.array([4, 7, 10, 13]).T + 2
    initial_w = 1.5
    initial_b = 1
    dj_dw, dj_db = target(x, y, initial_w, initial_b)
    #assert dj_dw.shape == initial_w.shape, f"Wrong shape for dj_dw. {dj_dw} != {initial_w.shape}"
    assert dj_db == -2, f"情况1: dj_db错误: {dj_db} != -2"
    assert np.allclose(dj_dw, -10.0), f"情况1: dj_dw错误: {dj_dw} != -10.0"   
    
    print("\033[92m所有测试通过!")
    

