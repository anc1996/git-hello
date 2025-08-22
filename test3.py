import numpy as np
import time

def matrix_add(A, B):
    """矩阵加法"""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def matrix_subtract(A, B):
    """矩阵减法"""
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def split_matrix(A):
    """将矩阵分成四个子矩阵"""
    n = len(A)
    mid = n // 2
    
    A11 = [[A[i][j] for j in range(mid)] for i in range(mid)]
    A12 = [[A[i][j] for j in range(mid, n)] for i in range(mid)]
    A21 = [[A[i][j] for j in range(mid)] for i in range(mid, n)]
    A22 = [[A[i][j] for j in range(mid, n)] for i in range(mid, n)]
    
    return A11, A12, A21, A22

def combine_matrices(C11, C12, C21, C22):
    """将四个子矩阵组合成一个矩阵"""
    n = len(C11)
    C = [[0 for _ in range(2*n)] for _ in range(2*n)]
    
    for i in range(n):
        for j in range(n):
            C[i][j] = C11[i][j]
            C[i][j+n] = C12[i][j]
            C[i+n][j] = C21[i][j]
            C[i+n][j+n] = C22[i][j]
    
    return C

def strassen_multiply(A, B):
    """
    Strassen 矩阵乘法算法
    输入: 两个 n×n 矩阵 A 和 B (n 必须是 2 的幂)
    输出: 矩阵乘积 C = A × B
    """
    n = len(A)
    
    # 基础情况：1×1 矩阵
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    
    # 如果矩阵大小不是 2 的幂，用传统方法
    if n % 2 != 0:
        return traditional_multiply(A, B)
    
    # 分割矩阵
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)
    
    # 计算 Strassen 的 7 个乘积
    P1 = strassen_multiply(A11, matrix_subtract(B12, B22))
    P2 = strassen_multiply(matrix_add(A11, A12), B22)
    P3 = strassen_multiply(matrix_add(A21, A22), B11)
    P4 = strassen_multiply(A22, matrix_subtract(B21, B11))
    P5 = strassen_multiply(matrix_add(A11, A22), matrix_add(B11, B22))
    P6 = strassen_multiply(matrix_subtract(A12, A22), matrix_add(B21, B22))
    P7 = strassen_multiply(matrix_subtract(A11, A21), matrix_add(B11, B12))
    
    # 计算结果矩阵的四个部分
    C11 = matrix_add(matrix_subtract(matrix_add(P5, P4), P2), P6)
    C12 = matrix_add(P1, P2)
    C21 = matrix_add(P3, P4)
    C22 = matrix_subtract(matrix_subtract(matrix_add(P5, P1), P3), P7)
    
    # 组合结果
    return combine_matrices(C11, C12, C21, C22)

def traditional_multiply(A, B):
    """传统矩阵乘法算法"""
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    
    return C

def pad_matrix(A, target_size):
    """将矩阵填充到目标大小（2的幂）"""
    n = len(A)
    padded = [[0 for _ in range(target_size)] for _ in range(target_size)]
    
    for i in range(n):
        for j in range(n):
            padded[i][j] = A[i][j]
    
    return padded

def unpad_matrix(A, original_size):
    """从填充矩阵中提取原始大小的矩阵"""
    return [[A[i][j] for j in range(original_size)] for i in range(original_size)]

def strassen_multiply_general(A, B):
    """
    通用的 Strassen 矩阵乘法（处理任意大小的矩阵）
    输入: 两个矩阵 A 和 B
    输出: 矩阵乘积 C = A × B
    """
    m, n = len(A), len(A[0])
    p, q = len(B), len(B[0])
    
    if n != p:
        raise ValueError("矩阵维度不匹配，无法相乘")
    
    # 找到最小的 2 的幂来容纳矩阵
    max_dim = max(m, n, q)
    target_size = 1
    while target_size < max_dim:
        target_size *= 2
    
    # 填充矩阵
    A_padded = pad_matrix(A, target_size)
    B_padded = pad_matrix(B, target_size)
    
    # 使用 Strassen 算法计算
    C_padded = strassen_multiply(A_padded, B_padded)
    
    # 提取结果
    return unpad_matrix(C_padded, m)

def print_matrix(A, name="矩阵"):
    """打印矩阵"""
    print(f"{name}:")
    for row in A:
        print(" ".join(f"{val:6}" for val in row))
    print()

def test_strassen():
    """测试 Strassen 算法"""
    print("=== Strassen 矩阵乘法算法测试 ===\n")
    
    # 测试用例 1: 2×2 矩阵
    print("测试用例 1: 2×2 矩阵")
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    
    print_matrix(A, "矩阵 A")
    print_matrix(B, "矩阵 B")
    
    # 使用 Strassen 算法
    C_strassen = strassen_multiply(A, B)
    print_matrix(C_strassen, "Strassen 结果")
    
    # 使用传统算法验证
    C_traditional = traditional_multiply(A, B)
    print_matrix(C_traditional, "传统算法结果")
    
    # 验证结果
    print(f"结果是否一致: {C_strassen == C_traditional}\n")
    
    # 测试用例 2: 4×4 矩阵
    print("测试用例 2: 4×4 矩阵")
    A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    B = [[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]]
    
    print_matrix(A, "矩阵 A")
    print_matrix(B, "矩阵 B")
    
    C_strassen = strassen_multiply(A, B)
    print_matrix(C_strassen, "Strassen 结果")
    
    C_traditional = traditional_multiply(A, B)
    print_matrix(C_traditional, "传统算法结果")
    
    print(f"结果是否一致: {C_strassen == C_traditional}\n")
    
    # 测试用例 3: 任意大小矩阵
    print("测试用例 3: 3×3 矩阵（使用通用版本）")
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    
    print_matrix(A, "矩阵 A")
    print_matrix(B, "矩阵 B")
    
    C_strassen = strassen_multiply_general(A, B)
    print_matrix(C_strassen, "Strassen 通用版本结果")
    
    C_traditional = traditional_multiply(A, B)
    print_matrix(C_traditional, "传统算法结果")
    
    print(f"结果是否一致: {C_strassen == C_traditional}\n")

def performance_comparison():
    """性能比较"""
    print("=== 性能比较 ===\n")
    
    sizes = [4, 8, 16, 32]
    
    for size in sizes:
        print(f"矩阵大小: {size}×{size}")
        
        # 生成随机矩阵
        A = [[np.random.randint(1, 10) for _ in range(size)] for _ in range(size)]
        B = [[np.random.randint(1, 10) for _ in range(size)] for _ in range(size)]
        
        # 测试传统算法
        start_time = time.time()
        C_traditional = traditional_multiply(A, B)
        traditional_time = time.time() - start_time
        
        # 测试 Strassen 算法
        start_time = time.time()
        C_strassen = strassen_multiply(A, B)
        strassen_time = time.time() - start_time
        
        # 验证结果
        is_correct = C_strassen == C_traditional
        
        print(f"  传统算法时间: {traditional_time:.6f} 秒")
        print(f"  Strassen 时间: {strassen_time:.6f} 秒")
        print(f"  加速比: {traditional_time/strassen_time:.2f}x")
        print(f"  结果正确: {is_correct}")
        print()

if __name__ == "__main__":
    # 运行测试
    test_strassen()
    
    # 运行性能比较
    performance_comparison()
    
    print("=== 算法说明 ===")
    print("Strassen 矩阵乘法算法:")
    print("1. 时间复杂度: O(n^2.807) vs 传统算法的 O(n^3)")
    print("2. 使用分治策略，将矩阵分成四个子矩阵")
    print("3. 通过巧妙的数学技巧，将 8 次乘法减少到 7 次")
    print("4. 适用于大型矩阵，对于小矩阵可能因为递归开销而较慢")
    print("5. 要求矩阵大小是 2 的幂，否则需要填充")
    print("")