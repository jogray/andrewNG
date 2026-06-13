import numpy as np

def compute_entropy_test(target):
    y = np.array([1] * 10)
    result = target(y)
    
    assert result == 0, "全为 1 的数组熵必须为 0"
    
    y = np.array([0] * 10)
    result = target(y)
    
    assert result == 0, "全为 0 的数组熵必须为 0"
    
    y = np.array([0] * 12 + [1] * 12)
    result = target(y)
    
    assert result == 1, "1 和 0 数量相同时熵必须为 1"
    
    y = np.array([1, 0, 1, 0, 1, 1, 1, 0, 1])
    assert np.isclose(target(y), 0.918295, atol=1e-6), "值错误。应为 0 到 1 之间的某个值"
    assert np.isclose(target(-y + 1), target(y), atol=1e-6), "值错误"
    
    print("\033[92m 所有测试通过。")

def split_dataset_test(target):

    # Case 1
    X = np.array([[1, 0], 
         [1, 0], 
         [1, 1], 
         [0, 0], 
         [0, 1]])
    X_t = np.array([[0, 1, 0, 1, 0]])
    X = np.concatenate((X, X_t.T), axis=1)

    left, right = target(X, list(range(5)), 2)
    expected = {'left': np.array([1, 3]),
                'right': np.array([0, 2, 4])}

    assert type(left) == list, f"左侧类型错误。期望: list，得到: {type(left)}"
    assert type(right) == list, f"右侧类型错误。期望: list，得到: {type(right)}"
    
    assert type(left[0]) == int, f"左侧列表元素类型错误。期望: int，得到: {type(left[0])}"
    assert type(right[0]) == int, f"右侧列表元素类型错误。期望: number，得到: {type(right[0])}"
    
    assert len(left) == 2, f"左侧必须有 2 个元素，但得到: {len(left)}"
    assert len(right) == 3, f"右侧必须有 3 个元素，但得到: {len(right)}"

    assert np.allclose(right, expected['right']), f"右侧值错误。期望: { expected['right']} \n得到: {right}"
    assert np.allclose(left, expected['left']), f"左侧值错误。期望: { expected['left']} \n得到: {left}"


    # Case 2
    X = np.array([[0, 1], 
         [1, 1], 
         [1, 1], 
         [0, 0], 
         [1, 0]])
    X_t = np.array([[0, 1, 0, 1, 0]])
    X = np.concatenate((X_t.T, X), axis=1)

    left, right = target(X, list(range(5)), 0)
    expected = {'left': np.array([1, 3]),
                'right': np.array([0, 2, 4])}


    assert len(left) == 2, f"左侧必须有 2 个元素，但得到: {len(left)}" 
    assert len(right) == 3, f"右侧必须有 3 个元素，但得到: {len(right)}"
    assert np.allclose(right, expected['right']) and np.allclose(left, expected['left']), f"目标在索引 0 时值错误。"


    # Case 3
    X = (np.random.rand(11, 3) > 0.5) * 1 # Just random binary numbers
    X_t = np.array([[0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0]])
    X = np.concatenate((X, X_t.T), axis=1)

    left, right = target(X, [1, 2, 3, 6, 7, 9, 10], 3)
    expected = {'left': np.array([1, 3, 6]),
                'right': np.array([2, 7, 9, 10])}

    assert np.allclose(right, expected['right']) and np.allclose(left, expected['left']), f"目标在索引 0 时值错误。 \n期望: {expected} \n得到: \{left:{left}, 'right': {right}\}"
 
    
    print("\033[92m All tests passed.")

def compute_information_gain_test(target):
    X = np.array([[1, 0], 
         [1, 0], 
         [1, 0], 
         [0, 0], 
         [0, 1]])
    
    y = np.array([[0, 0, 0, 0, 0]]).T
    node_indexes = list(range(5))

    result1 = target(X, y, node_indexes, 0)
    result2 = target(X, y, node_indexes, 0)
    
    assert result1 == 0 and result2 == 0, f"当目标变量纯净时，信息增益必须为 0。得到 {result1} 和 {result2}"
    
    y = np.array([[0, 1, 0, 1, 0]]).T
    node_indexes = list(range(5))
    
    result = target(X, y, node_indexes, 0)
    assert np.isclose(result, 0.019973, atol=1e-6), f"信息增益错误。期望 {0.019973}，得到: {result}"
    
    result = target(X, y, node_indexes, 1)
    assert np.isclose(result, 0.170951, atol=1e-6), f"信息增益错误。期望 {0.170951}，得到: {result}"

    node_indexes = list(range(4))
    result = target(X, y, node_indexes, 0)
    assert np.isclose(result, 0.311278, atol=1e-6), f"信息增益错误。期望 {0.311278}，得到: {result}"

    result = target(X, y, node_indexes, 1)
    assert np.isclose(result, 0, atol=1e-6), f"信息增益错误。期望 {0.0}，得到: {result}"

    print("\033[92m 所有测试通过。")

def get_best_split_test(target):
    X = np.array([[1, 0], 
         [1, 0], 
         [1, 0], 
         [0, 0], 
         [0, 1]])

    y = np.array([[0, 0, 0, 0, 0]]).T
    node_indexes = list(range(5))

    result = target(X, y, node_indexes)
    
    assert result == -1, f"当目标变量纯净时，没有最佳分割。期望 -1，得到 {result}"
    
    y = X[:,0]
    result = target(X, y, node_indexes)
    assert result == 0, f"如果目标与其他特征完全相关，该特征必须是最佳分割。期望 0，得到 {result}"
    y = X[:,1]
    result = target(X, y, node_indexes)
    assert result == 1, f"如果目标与其他特征完全相关，该特征必须是最佳分割。期望 1，得到 {result}"

    y = 1 - X[:,0]
    result = target(X, y, node_indexes)
    assert result == 0, f"如果目标与其他特征完全相关，该特征必须是最佳分割。期望 0，得到 {result}"

    y = np.array([[0, 1, 0, 1, 0]]).T
    result = target(X, y, node_indexes)
    assert result == 1, f"结果错误。期望 1，得到 {result}"

    y = np.array([[0, 1, 0, 1, 0]]).T    
    node_indexes = [2, 3, 4]
    result = target(X, y, node_indexes)
    assert result == 0, f"结果错误。期望 0，得到 {result}"

    n_samples = 100
    X0 = np.array([[1] * n_samples])
    X1 = np.array([[0] * n_samples])
    X2 = (np.random.rand(1, 100) > 0.5) * 1
    X3 = np.array([[1] * int(n_samples / 2) + [0] * int(n_samples / 2)])
    
    y = X2.T
    node_indexes = list(range(20, 80))
    X = np.array([X0, X1, X2, X3]).T.reshape(n_samples, 4)
    result = target(X, y, node_indexes)
    
    assert result == 2, f"结果错误。期望 2，得到 {result}"
    
    y = X0.T
    result = target(X, y, node_indexes)
    assert result == -1, f"当目标变量纯净时，没有最佳分割。期望 -1，得到 {result}"
    print("\033[92m 所有测试通过。")
