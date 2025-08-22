import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    简单的前馈神经网络实现
    
    支持多层网络，使用sigmoid激活函数和均方误差损失函数
    """
    
    def __init__(self, layers):
        """
        初始化神经网络
        
        参数:
        layers: 列表，包含每层神经元的数量
               例如: [2, 3, 1] 表示输入层2个神经元，隐藏层3个神经元，输出层1个神经元
        """
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = []
        self.biases = []
        
        # 初始化权重和偏置
        for i in range(self.num_layers - 1):
            # 使用He初始化方法
            w = np.random.randn(layers[i + 1], layers[i]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((layers[i + 1], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, z):
        """sigmoid激活函数"""
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """sigmoid函数的导数"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward_propagation(self, X):
        """
        前向传播
        
        参数:
        X: 输入数据，形状为 (input_size, batch_size)
        
        返回:
        activations: 每层的激活值
        z_values: 每层的z值（激活前的值）
        """
        activations = [X]
        z_values = []
        
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            z_values.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        
        return activations, z_values
    
    def backward_propagation(self, X, y, activations, z_values, learning_rate=0.1):
        """
        反向传播算法
        
        参数:
        X: 输入数据
        y: 真实标签
        activations: 前向传播得到的激活值
        z_values: 前向传播得到的z值
        learning_rate: 学习率
        """
        m = X.shape[1]  # 批次大小
        
        # 计算输出层的误差
        delta = activations[-1] - y
        
        # 反向传播误差
        for i in range(self.num_layers - 2, -1, -1):
            # 计算权重梯度
            dW = np.dot(delta, activations[i].T) / m
            db = np.sum(delta, axis=1, keepdims=True) / m
            
            # 更新权重和偏置
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
            
            # 计算下一层的误差（除了输入层）
            if i > 0:
                delta = np.dot(self.weights[i].T, delta) * self.sigmoid_derivative(z_values[i-1])
    
    def train(self, X, y, epochs, learning_rate=0.1, batch_size=None):
        """
        训练神经网络
        
        参数:
        X: 输入数据，形状为 (input_size, samples)
        y: 真实标签，形状为 (output_size, samples)
        epochs: 训练轮数
        learning_rate: 学习率
        batch_size: 批次大小，None表示使用全部数据
        
        返回:
        losses: 每轮的损失值
        """
        losses = []
        
        for epoch in range(epochs):
            if batch_size is None:
                # 使用全部数据
                activations, z_values = self.forward_propagation(X)
                loss = np.mean((activations[-1] - y) ** 2)
                self.backward_propagation(X, y, activations, z_values, learning_rate)
            else:
                # 小批次训练
                total_loss = 0
                num_batches = X.shape[1] // batch_size
                
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size
                    
                    X_batch = X[:, start_idx:end_idx]
                    y_batch = y[:, start_idx:end_idx]
                    
                    activations, z_values = self.forward_propagation(X_batch)
                    loss = np.mean((activations[-1] - y_batch) ** 2)
                    total_loss += loss
                    
                    self.backward_propagation(X_batch, y_batch, activations, z_values, learning_rate)
                
                loss = total_loss / num_batches
            
            losses.append(loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        return losses
    
    def predict(self, X):
        """
        预测
        
        参数:
        X: 输入数据
        
        返回:
        预测结果
        """
        activations, _ = self.forward_propagation(X)
        return activations[-1]
    
    def accuracy(self, X, y, threshold=0.5):
        """
        计算准确率（用于分类问题）
        
        参数:
        X: 输入数据
        y: 真实标签
        threshold: 分类阈值
        
        返回:
        准确率
        """
        predictions = self.predict(X)
        predictions = (predictions > threshold).astype(int)
        return np.mean(predictions == y)

def generate_xor_data():
    """生成XOR问题的数据"""
    X = np.array([[0, 0, 1, 1],
                   [0, 1, 0, 1]])
    y = np.array([[0, 1, 1, 0]])
    return X, y

def generate_linear_data(n_samples=100):
    """生成线性回归数据"""
    np.random.seed(42)
    X = np.random.randn(1, n_samples) * 2
    y = 2 * X + 1 + np.random.randn(1, n_samples) * 0.1
    return X, y

def test_neural_network():
    """测试神经网络算法"""
    print("=== 神经网络算法测试 ===\n")
    
    # 测试1: XOR问题（分类）
    print("测试1: XOR问题（分类）")
    X_xor, y_xor = generate_xor_data()
    print(f"输入数据形状: {X_xor.shape}")
    print(f"标签形状: {y_xor.shape}")
    
    # 创建神经网络: 2个输入，4个隐藏神经元，1个输出
    nn_xor = NeuralNetwork([2, 4, 1])
    
    # 训练
    print("\n开始训练XOR网络...")
    losses_xor = nn_xor.train(X_xor, y_xor, epochs=1000, learning_rate=0.5)
    
    # 测试
    predictions_xor = nn_xor.predict(X_xor)
    accuracy_xor = nn_xor.accuracy(X_xor, y_xor)
    
    print(f"\nXOR问题结果:")
    print(f"预测值: {predictions_xor.flatten()}")
    print(f"真实值: {y_xor.flatten()}")
    print(f"准确率: {accuracy_xor:.2%}")
    
    # 测试2: 线性回归
    print("\n" + "="*50)
    print("测试2: 线性回归")
    X_linear, y_linear = generate_linear_data(200)
    print(f"输入数据形状: {X_linear.shape}")
    print(f"标签形状: {y_linear.shape}")
    
    # 创建神经网络: 1个输入，3个隐藏神经元，1个输出
    nn_linear = NeuralNetwork([1, 3, 1])
    
    # 训练
    print("\n开始训练线性回归网络...")
    losses_linear = nn_linear.train(X_linear, y_linear, epochs=500, learning_rate=0.1)
    
    # 测试
    predictions_linear = nn_linear.predict(X_linear)
    mse = np.mean((predictions_linear - y_linear) ** 2)
    
    print(f"\n线性回归结果:")
    print(f"均方误差: {mse:.6f}")
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    # 绘制XOR损失
    plt.subplot(1, 3, 1)
    plt.plot(losses_xor)
    plt.title('XOR问题训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 绘制线性回归损失
    plt.subplot(1, 3, 2)
    plt.plot(losses_linear)
    plt.title('线性回归训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 绘制线性回归结果
    plt.subplot(1, 3, 3)
    plt.scatter(X_linear.flatten(), y_linear.flatten(), alpha=0.6, label='真实数据')
    plt.scatter(X_linear.flatten(), predictions_linear.flatten(), alpha=0.6, label='预测值')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('线性回归结果')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 测试3: 网络结构信息
    print("\n" + "="*50)
    print("测试3: 网络结构信息")
    print(f"XOR网络层数: {nn_xor.num_layers}")
    print(f"XOR网络结构: {nn_xor.layers}")
    print(f"权重形状: {[w.shape for w in nn_xor.weights]}")
    print(f"偏置形状: {[b.shape for b in nn_xor.biases]}")

def demonstrate_custom_function():
    """演示自定义函数拟合"""
    print("\n" + "="*50)
    print("演示: 拟合自定义函数 f(x) = sin(x) + 0.5")
    
    # 生成数据
    X = np.linspace(-3, 3, 100).reshape(1, -1)
    y = np.sin(X) + 0.5
    
    # 创建网络
    nn_custom = NeuralNetwork([1, 10, 5, 1])
    
    # 训练
    print("开始训练自定义函数网络...")
    losses_custom = nn_custom.train(X, y, epochs=1000, learning_rate=0.01)
    
    # 预测
    predictions_custom = nn_custom.predict(X)
    mse_custom = np.mean((predictions_custom - y) ** 2)
    
    print(f"拟合结果 - 均方误差: {mse_custom:.6f}")
    
    # 可视化
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses_custom)
    plt.title('自定义函数训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(X.flatten(), y.flatten(), 'b-', label='真实函数', linewidth=2)
    plt.plot(X.flatten(), predictions_custom.flatten(), 'r--', label='神经网络拟合', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('函数拟合结果')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_neural_network()
    demonstrate_custom_function()
