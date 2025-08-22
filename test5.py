# -*- coding: utf-8 -*-

def fibonacci_dp(n):
    """
    使用动态规划 (自底向上) 计算斐波那契数列的第 n 项。

    动态规划思想：
    1. 定义状态：dp[i] 表示斐波那契数列的第 i 项的值。
    2. 状态转移方程：dp[i] = dp[i-1] + dp[i-2]。
    3. 初始化：dp[0] = 0, dp[1] = 1。
    4. 遍历顺序：从 i = 2 开始，一直计算到 n。

    Args:
        n (int): 需要计算的斐波那契数列的项数 (非负整数)。

    Returns:
        int: 斐波那契数列的第 n 项的值。
        Returns -1 if input is negative.
    """
    # --- 输入检查 ---
    if not isinstance(n, int) or n < 0:
        print("输入错误：请输入一个非负整数。")
        return -1

    # --- 基础情况处理 ---
    if n <= 1:
        return n

    # --- 动态规划过程 ---
    
    # 1. 初始化一个列表 (或数组) 来存储计算结果。
    # 我们只需要 n+1 个位置来存储从 0 到 n 的所有结果。
    dp = [0] * (n + 1)
    
    # 2. 设置初始值，这是递推的起点。
    dp[0] = 0
    dp[1] = 1
    
    # 3. 循环计算，从第 2 项开始，直到第 n 项。
    # 这里的每一步都利用了前面已经计算好的结果。
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
        
    # 4. 返回最终结果。
    # dp[n] 中存储的就是我们想要的目标值。
    return dp[n]

# --- 主程序：定义输入并调用函数 ---
if __name__ == "__main__":
    # 定义输入：我们想计算斐波那契数列的第 10 项
    input_number = 10
    
    print(f"输入: n = {input_number}")
    
    # 调用函数计算结果
    output_result = fibonacci_dp(input_number)
    
    # 打印输出
    if output_result != -1:
        print(f"输出: 斐波那契数列的第 {input_number} 项是 {output_result}")
    
    print("-" * 20)

    # 另一个例子
    input_number_2 = 7
    print(f"输入: n = {input_number_2}")
    output_result_2 = fibonacci_dp(input_number_2)
    if output_result_2 != -1:
        print(f"输出: 斐波那契数列的第 {input_number_2} 项是 {output_result_2}")
    print(f"结束")
    
