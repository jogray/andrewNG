import numpy as np

def compute_centroids_test(target):
    # 使用3个质心
    X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, 1],
                  [-1, 1.5], [2.5, 1.5], [-1.1, -1.7], [-1.6, 1.2]])
    idx = np.array([1, 1, 1, 0, 0, 0, 2])
    K = 3
    centroids = target(X, idx, K)
    expected_centroids = np.array([[0.13333333,  0.43333333],
                                   [-1.33333333, -0.5      ],
                                   [-1.6,        1.2       ]])
    
    assert type(centroids) == np.ndarray, "类型错误"
    assert centroids.shape == (K, X.shape[1]), f"形状错误。期望: {(len(X),)} 得到: {idx.shape}"
    assert np.allclose(centroids, expected_centroids), f"值错误。期望: {expected_centroids}, 得到: {centroids}"
    
    X = np.array([[2, 2.5], [2.5, 2.5], [-1.5, -1.5],
                  [2, 2], [-1.5, -1], [-1, -1]])
    idx = np.array([0, 0, 1, 0, 1, 1])
    K = 2
    centroids = target(X, idx, K)
    expected_centroids = np.array([[[ 2.16666667,  2.33333333],
                                    [-1.33333333, -1.16666667]]])
    
    assert type(centroids) == np.ndarray, "类型错误"
    assert centroids.shape == (K, X.shape[1]), f"形状错误。期望: {(len(X),)} 得到: {idx.shape}"
    assert np.allclose(centroids, expected_centroids), f"值错误。期望: {expected_centroids}, 得到: {centroids}"
    
    print("\033[92m所有测试通过！")
    
def find_closest_centroids_test(target):
    # 使用2个质心
    X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, -1],
                  [2, 2],[2.5, 2.5],[2, 2.5]])
    initial_centroids = np.array([[-1, -1], [2, 2]])
    idx = target(X, initial_centroids)
    
    assert type(idx) == np.ndarray, "类型错误"
    assert idx.shape == (len(X),), f"形状错误。期望: {(len(X),)} 得到: {idx.shape}"
    assert np.allclose(idx, [0, 0, 0, 1, 1, 1]), "值错误"
    
    # 使用3个质心
    X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, 1],
                  [-1, 1.5], [2.5, 1.5], [2, 2]])
    initial_centroids = np.array([[2.5, 2], [-1, -1], [-1.5, 1.]])
    idx = target(X, initial_centroids)
    
    assert type(idx) == np.ndarray, "类型错误"
    assert idx.shape == (len(X),), f"形状错误。期望: {(len(X),)} 得到: {idx.shape}"
    assert np.allclose(idx, [1, 1, 2, 2, 0, 0]), f"值错误。期望 {[2, 2, 0, 0, 1, 1]}, 得到: {idx}"
    
    # 使用3个质心
    X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, 1],
                  [-1, 1.5], [2.5, 1.5], [-1.1, -1.7], [-1.6, 1.2]])
    initial_centroids = np.array([[2.5, 2], [-1, -1], [-1.5, 1.]])
    idx = target(X, initial_centroids)
    
    assert type(idx) == np.ndarray, "类型错误"
    assert idx.shape == (len(X),), f"形状错误。期望: {(len(X),)} 得到: {idx.shape}"
    assert np.allclose(idx, [1, 1, 2, 2, 0, 1, 2]), f"值错误。期望 {[2, 2, 0, 0, 1, 1]}, 得到: {idx}"
    
    print("\033[92m所有测试通过！")