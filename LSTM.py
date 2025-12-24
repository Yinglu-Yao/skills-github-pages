# ==============================
# 0. 导入库
# ==============================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 固定随机种子（可复现）
torch.manual_seed(42)
np.random.seed(42)


# ==============================
# 1. 构造原始数据集（正弦波）
# ==============================
time_steps = np.linspace(0, 100, 1000)
data = np.sin(time_steps)

# 可视化原始数据
plt.figure(figsize=(10, 3))
plt.plot(data)
plt.title("Raw Time Series Data")
plt.show()


# ==============================
# 2. 滑动窗口：原始数据 → 样本
# ==============================
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i + seq_len])
        ys.append(data[i + seq_len])
    return np.array(xs), np.array(ys)


SEQ_LEN = 20
X, y = create_sequences(data, SEQ_LEN)

# ==============================
# 3. 转成 LSTM 需要的张量格式
# ==============================
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 增加特征维度（input_size = 1）
X = X.unsqueeze(-1)   # (samples, seq_len, 1)
y = y.unsqueeze(-1)   # (samples, 1)

# ==============================
# 4. 划分训练 / 测试集
# ==============================
train_size = int(len(X) * 0.8)

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]


# ==============================
# 5. 定义 LSTM 模型
# ==============================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步
        return out


# ==============================
# 6. 初始化模型 & 训练参数
# ==============================
model = LSTMModel(
    input_size=1,
    hidden_size=64,
    num_layers=1,
    output_size=1
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ==============================
# 7. 训练模型
# ==============================
EPOCHS = 200
loss_history = []
model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")


# ==============================
# 8. 测试 & 预测
# ==============================
model.eval()
with torch.no_grad():
    predictions = model(X_test)

# 转成 numpy 方便画图
predictions = predictions.squeeze().numpy()
y_test_np = y_test.squeeze().numpy()


# ==============================
# 9. 可视化预测结果
# ==============================
plt.figure(figsize=(10, 4))
plt.plot(y_test_np, label="True")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("LSTM Time Series Prediction")
plt.show()
# ==============================
# 10. 可视化 Loss 曲线
# ==============================
plt.figure(figsize=(5, 4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()

