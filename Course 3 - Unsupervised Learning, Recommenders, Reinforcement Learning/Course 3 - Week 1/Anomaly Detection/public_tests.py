import numpy as np
import random

def select_threshold_test(target):
    p_val = np.array([i / 100 for i in range(30)])
    y_val = np.array([1] * 5 + [0] * 25)
    
    best_epsilon, best_F1 = target(y_val, p_val)
    assert np.isclose(best_epsilon, 0.04, atol=0.3 / 1000), f"best_epsilon错误。期望: {0.04} 得到: {best_epsilon}"
    assert best_F1 == 1, f"best_F1错误。期望: 1 得到: {best_F1}"
    
    y_val = np.array([1] * 5 + [0] * 25)
    y_val[2] = 0 # 引入噪声
    best_epsilon, best_F1 = target(y_val, p_val)
    assert np.isclose(best_epsilon, 0.04, atol=0.3 / 1000), f"best_epsilon错误。期望: {0.04} 得到: {best_epsilon}"
    assert np.isclose(best_F1, 0.8888888), f"best_F1错误。期望: 0.8888888 得到: {best_F1}"
    
    p_val = np.array([i / 1000 for i in range(50)])
    y_val = np.array([1] * 8 + [0] * 42)
    y_val[5] = 0
    index = [*range(50)]
    random.shuffle(index)
    p_val = p_val[index]
    y_val = y_val[index]

    best_epsilon, best_F1 = target(y_val, p_val)
    assert np.isclose(best_epsilon, 0.007, atol=0.05 / 1000), f"best_epsilon错误。期望: {0.0070070} 得到: {best_epsilon}"
    assert np.isclose(best_F1, 0.933333333), f"best_F1错误。期望: 0.933333333 得到: {best_F1}"
    print("\033[92m所有测试通过！")
    
def estimate_gaussian_test(target):
    np.random.seed(273)
    
    X = np.array([[1, 1, 1], 
                  [2, 2, 2], 
                  [3, 3, 3]]).T
    
    mu, var = target(X)
    
    assert type(mu) == np.ndarray, f"mu类型错误。期望: {np.ndarray} 得到: {type(mu)}"
    assert type(var) == np.ndarray, f"var类型错误。期望: {np.ndarray} 得到: {type(var)}"
    
    assert mu.shape == (X.shape[1],), f"mu形状错误。期望: {(X.shape[1],)} 得到: {mu.shape}"
    assert type(var) == np.ndarray, f"var形状错误。期望: {(X.shape[1],)} 得到: {var.shape}"
    
    assert np.allclose(mu, [1., 2., 3.]), f"mu值错误。期望: {[1, 2, 3]} 得到: {mu}"
    assert np.allclose(var, [0., 0., 0.]), f"var值错误。期望: {[0, 0, 0]} 得到: {var}"
    
    X = np.array([[1, 2, 3], 
                  [2, 4, 6], 
                  [3, 6, 9]]).T
    
    mu, var = target(X)
    
    assert type(mu) == np.ndarray, f"mu类型错误。期望: {np.ndarray} 得到: {type(mu)}"
    assert type(var) == np.ndarray, f"var类型错误。期望: {np.ndarray} 得到: {type(var)}"
    
    assert mu.shape == (X.shape[1],), f"mu形状错误。期望: {(X.shape[1],)} 得到: {mu.shape}"
    assert type(var) == np.ndarray, f"var形状错误。期望: {(X.shape[1],)} 得到: {var.shape}"
    
    assert np.allclose(mu, [2., 4., 6.]), f"mu值错误。期望: {[2., 4., 6.]} 得到: {mu}"
    assert np.allclose(var, [2. / 3, 8. / 3., 18. / 3.]), f"var值错误。期望: {[2. / 3, 8. / 3., 18. / 3.]} 得到: {var}"
    
    
    m = 500
    X = np.array([np.random.normal(0, 1, m), 
                  np.random.normal(1, 2, m), 
                  np.random.normal(3, 1.5, m)]).T
    
    mu, var = target(X)
    
    assert type(mu) == np.ndarray, f"mu类型错误。期望: {np.ndarray} 得到: {type(mu)}"
    assert type(var) == np.ndarray, f"var类型错误。期望: {np.ndarray} 得到: {type(var)}"
    
    assert mu.shape == (X.shape[1],), f"mu形状错误。期望: {(X.shape[1],)} 得到: {mu.shape}"
    assert type(var) == np.ndarray, f"var形状错误。期望: {(X.shape[1],)} 得到: {var.shape}"
    
    assert np.allclose(mu, [0., 1., 3.], atol=0.2), f"mu值错误。期望: {[0, 1, 3]} 得到: {mu}"
    assert np.allclose(var, np.square([1., 2., 1.5]), atol=0.2), f"var值错误。期望: {np.square([1., 2., 1.5])} 得到: {var}"
    
    print("\033[92m所有测试通过！")