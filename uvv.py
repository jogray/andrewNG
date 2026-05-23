import torch
import tensorflow as tf
import jax.numpy as jnp
import jax 
import time
import gc
import numpy as np
import matplotlib.pyplot as plt
records = []
import time

def gpu_stress_test(lib, intensity: int, max_mem_gb: float, rounds: int = 5):
    """
    基于鸭子类型的通用显卡/算力压力测试函数
    
    :param lib: 传入的库对象 (torch, tf, jax.numpy 或 numpy/cupy)
    :param intensity: 压力强度，决定矩阵乘法的大小 (例如 5000, 10000)
    :param max_mem_gb: 目标最大显存占用 (GB)，用于动态调整辅助矩阵大小
    :param rounds: 测试轮数
    :return: 每轮计算耗时的列表 (秒)
    """
    print(f"正在使用库 [{lib.__name__}] 进行压力测试...")
    
    # 1. 根据 max_mem_gb 估算需要创建的辅助大矩阵大小，用于强行霸占显存
    # 一个 float32 占用 4 字节。大矩阵大小为 N x N
    # N^2 * 4 = max_mem_gb * 1024^3
    if max_mem_gb > 0:
        matrix_size = int((max_mem_gb * (1024**3) / 4) ** 0.5)
        print(f"-> 正在分配约 {max_mem_gb} GB 的显存/内存...")
        try:
            # 鸭子类型：各大库都支持 lib.ones((shape))
            memory_holder = lib.ones((matrix_size, matrix_size))
        except Exception as e:
            print(f"显存分配失败（可能是超出了实际显存），错误信息: {e}")
            memory_holder = None
    else:
        memory_holder = None

    # 2. 创建用于高压密集计算的核心矩阵 (intensity x intensity)
    print(f"-> 创建核心计算矩阵: {intensity} x {intensity}")
    # 鸭子类型：各大库均支持 lib.ones 且支持通过 @ 或 matmul 计算
    a = lib.ones((intensity, intensity))
    b = lib.ones((intensity, intensity))
    
    # 针对需要显式同步的库（如 PyTorch GPU 或 JAX）进行特殊同步处理
    def sync_device():
        if hasattr(lib, "cuda") and hasattr(lib.cuda, "synchronize"):
            lib.cuda.synchronize()  # PyTorch 同步
        elif hasattr(a, "block_until_ready"):
            a.block_until_ready()  # JAX 异步计算同步

    # 预热一轮，防止把框架初始化的时间算进去
    sync_device()
    _ = a @ b
    sync_device()

    # 3. 开始高压计算轮询
    time_records = []
    print(f"-> 开始高压矩阵乘法测试，共 {rounds} 轮...")
    
    for r in range(rounds):
        sync_device()
        start_time = time.perf_counter()
        
        # 密集矩阵乘法，循环 10 次放大压力
        c = a
        for _ in range(10):
            c = c @ b  
            
        sync_device()
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        time_records.append(duration)
        print(f"   轮次 {r+1}/{rounds} 耗时: {duration:.4f} 秒")

    # 释放显存引用
    del memory_holder, a, b, c
    
    return time_records
# 确保目标设备是 GPU
try:
    if torch.cuda.is_available():
        # 技巧：通过偏函数或包装，让传入的 lib 默认在 GPU 上创建
        # 鸭子类型要求 lib.ones 能用，我们重写一个支持 cuda 的 ones
        class TorchGPUAdapter:
            __name__ = "torch (CUDA)"
            def ones(self, shape):
                return torch.ones(shape, device="cuda")
            def cuda(self):
                return torch.cuda

        records = gpu_stress_test(TorchGPUAdapter(), intensity=8000, max_mem_gb=4.0, rounds=5)
except ModuleNotFoundError:
    print("未导入pytorch，忽略PyTorch GPU测试。")
except Exception as e:
    print(f"PyTorch GPU测试失败，错误信息: {e}")

try:
    tf.config.list_physical_devices('GPU')  # 确保 TensorFlow 能看到 GPU
    # TensorFlow 在有 GPU 的环境下默认就会把操作分配到 GPU 上
    class TFGPUAdapter:
        __name__ = "tensorflow"
        def ones(self, shape):
            return tf.ones(shape)

    # 确保 TF 不会把显存吃满限制弹性，或直接运行
    records = gpu_stress_test(TFGPUAdapter(), intensity=8000, max_mem_gb=4.0, rounds=5)
except ModuleNotFoundError:
    print("未导入TensorFlow，忽略TensorFlow GPU测试。")
except Exception as e:
    print(f"TensorFlow GPU测试失败，错误信息: {e}")


try:
    jax.devices()  # 确保 JAX 能看到 GPU
    # JAX 默认在 GPU 可用时直接使用 GPU
    records = gpu_stress_test(jnp, intensity=8000, max_mem_gb=4.0, rounds=5)
except ModuleNotFoundError:
    print("未导入JAX，忽略JAX GPU测试。")
except Exception as e:
    print(f"JAX GPU测试失败，错误信息: {e}")